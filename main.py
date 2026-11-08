# =========================================================
# 0. SECCIÓN DE IMPORTS
# =========================================================
import asyncio
import io
import logging
import os
import re
import sqlite3
import threading
import uuid
import random
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from urllib.parse import urlparse

import discord
import requests
from discord.ext import commands
from flask import Flask, jsonify
from PIL import Image, UnidentifiedImageError
from crewai import Agent, Task, Crew, Process, LLM
from threading import Thread

# =========================================================
# 1. CONFIGURACIÓN
# =========================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("threads-bot")

DATABASE_PATH = os.getenv("DATABASE_PATH", "bot_state.sqlite3")
PORT = int(os.getenv("PORT", "10000"))

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MAX_IDEA_CHARS = 5000
MAX_POST_CHARS = 500  # Threads limita las publicaciones a 500 caracteres
PREVIEW_CHARS = 450
CREW_TIMEOUT = 120
HTTP_TIMEOUT = (5, 30)

THREADS_API_BASE = "https://graph.threads.net/v1.0"

executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="threads-worker")
app = Flask(__name__)

intents = discord.Intents.default()
intents.message_content = True


class ThreadsBot(commands.Bot):
    async def setup_hook(self):
        try:
            rows = list_drafts("PENDING", limit=10000)
            for row in rows:
                self.add_view(ApprovalView(row["id"]))
            log.info("Vistas persistentes registradas: %s", len(rows))
        except Exception:
            log.exception("No se pudieron registrar las vistas persistentes.")
            raise


bot = ThreadsBot(command_prefix="!", intents=intents)


def env_required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Falta la variable de entorno: {name}")
    return value


def authorized_users() -> set[int]:
    raw = env_required("DISCORD_ALLOWED_USER_IDS")
    try:
        ids = {int(item.strip()) for item in raw.split(",") if item.strip()}
    except ValueError as exc:
        raise RuntimeError(
            "DISCORD_ALLOWED_USER_IDS debe contener IDs numéricos separados por comas."
        ) from exc
    if not ids:
        raise RuntimeError("DISCORD_ALLOWED_USER_IDS no contiene IDs.")
    return ids


def is_authorized(user_id: int) -> bool:
    try:
        return user_id in authorized_users()
    except Exception:
        log.exception("No se pudo validar al usuario.")
        return False


def check_configuration() -> None:
    for name in (
        "DISCORD_BOT_TOKEN",
        "THREADS_ACCESS_TOKEN",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
    ):
        env_required(name)

    authorized_users()

    if not DATABASE_PATH:
        raise RuntimeError("DATABASE_PATH no puede estar vacío.")


# =========================================================
# 2. BASE DE DATOS
# =========================================================

@contextmanager
def db():
    conn = sqlite3.connect(DATABASE_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS drafts (
            id TEXT PRIMARY KEY,
            owner_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            image BLOB,
            status TEXT NOT NULL,
            post_id TEXT,
            error TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_drafts_status_created
        ON drafts(status, created_at)
        """)
        conn.execute("""
        UPDATE drafts
        SET status = 'UNKNOWN',
            error = 'Proceso reiniciado durante la publicación; verificar Threads',
            updated_at = CURRENT_TIMESTAMP
        WHERE status = 'PUBLISHING'
        """)


def save_draft(draft_id: str, owner_id: int, text: str, image: bytes | None) -> None:
    with db() as conn:
        conn.execute("""
        INSERT INTO drafts (id, owner_id, text, image, status)
        VALUES (?, ?, ?, ?, 'PENDING')
        """, (draft_id, owner_id, text, image))


def get_draft(draft_id: str):
    with db() as conn:
        return conn.execute(
            "SELECT * FROM drafts WHERE id = ?", (draft_id,)
        ).fetchone()


def claim_draft(draft_id: str) -> bool:
    with db() as conn:
        result = conn.execute("""
        UPDATE drafts
        SET status = 'PUBLISHING', error = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ? AND status = 'PENDING'
        """, (draft_id,))
        return result.rowcount == 1


def discard_draft(draft_id: str) -> bool:
    with db() as conn:
        result = conn.execute("""
        UPDATE drafts
        SET status = 'DISCARDED', updated_at = CURRENT_TIMESTAMP
        WHERE id = ? AND status = 'PENDING'
        """, (draft_id,))
        return result.rowcount == 1


def change_status(
    draft_id: str,
    status: str,
    post_id: str | None = None,
    error: str | None = None,
) -> None:
    safe_error = error[:1000] if error else None
    with db() as conn:
        conn.execute("""
        UPDATE drafts
        SET status = ?, post_id = COALESCE(?, post_id), error = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """, (status, post_id, safe_error, draft_id))


def list_drafts(status: str, limit: int = 20):
    with db() as conn:
        return conn.execute("""
        SELECT id, owner_id, text, status, post_id, error, created_at
        FROM drafts
        WHERE status = ?
        ORDER BY created_at DESC
        LIMIT ?
        """, (status, limit)).fetchall()


def database_ok() -> bool:
    try:
        with db() as conn:
            conn.execute("SELECT 1").fetchone()
            return True
    except Exception:
        log.exception("Falló el health check de SQLite.")
        return False


# =========================================================
# 3. FLASK: HEALTH CHECK & KEEP ALIVE
# =========================================================
from flask import Response

@app.get("/imagen/<draf_id>")
def obtener_imagen_draft(draft_id):
    draft = get_draft(draft_id)
    if not draft or not draft["image"]:
        return "Imagen no encontrada", 404
    return Response(draft["image"], mimetype="image/png")

@app.get("/")
def home():
    return "Bot de Threads activo", 200


@app.get("/live")
def live():
    return jsonify({"status": "alive"}), 200


@app.get("/health")
def health():
    db_ready = database_ok()
    discord_ready = bot.is_ready()
    healthy = db_ready and discord_ready
    return jsonify({
        "status": "ok" if healthy else "degraded",
        "database": "ok" if db_ready else "error",
        "discord": "ready" if discord_ready else "not_ready",
    }), (200 if healthy else 503)


def run_flask():
    app.run(host="0.0.0.0", port=PORT, use_reloader=False)


# =========================================================
# 4. VALIDACIÓN Y GENERACIÓN DE IMÁGENES
# =========================================================

def validate_image(data: bytes) -> bytes:
    if not data:
        raise ValueError("La imagen está vacía.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("La imagen supera el límite de 8 MiB.")

    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.width * img.height > MAX_IMAGE_PIXELS:
                raise ValueError("La imagen tiene demasiados píxeles.")
            img.verify()

        with Image.open(io.BytesIO(data)) as img:
            if img.format not in ("PNG", "JPEG", "WEBP"):
                raise ValueError("Formato permitido: PNG, JPEG o WEBP.")
            img.load()
            normalized = img.convert("RGBA" if "A" in img.getbands() else "RGB")
            output = io.BytesIO()
            normalized.save(output, format="PNG", optimize=True)
            result = output.getvalue()

        if len(result) > MAX_IMAGE_BYTES:
            raise ValueError("La imagen procesada supera 8 MiB.")
        return result

    except UnidentifiedImageError as exc:
        raise ValueError("El archivo no es una imagen válida.") from exc
    except Image.DecompressionBombError as exc:
        raise ValueError("La imagen excede los límites de seguridad.") from exc


def generar_imagen(idea: str) -> bytes | None:
    texto_limpio = idea.replace("\n"," ")[:250]
    prompt = f"Funny cartoon illustration, humorous comic art style representing: {texto_limpio}. Satirical visual joke, expressive funny characters, vibrant colorful cartoon, clean design, no text, no letters"
    try:
        response = requests.get(
            "https://image.pollinations.ai/prompt/"
            + requests.utils.quote(prompt, safe=""),
            params={
                "width": 1024,
                "height": 1024,
                "seed": random.randint(1, 1000000),
                "nologo": "true"
            },
            timeout=(5, 45),
        )
        response.raise_for_status()
        return validate_image(response.content)
    except Exception:
        log.warning("Pollinations AI no respondió a tiempo; continuando sin imagen.")
        return None


# =========================================================
# 5. PUBLICACIÓN EN THREADS (META GRAPH API)
# =========================================================

def obtener_threads_user_id(token: str) -> str:
    user_id = os.getenv("THREADS_USER_ID", "").strip()
    if user_id:
        return user_id

    response = requests.get(
        f"{THREADS_API_BASE}/me",
        params={"fields": "id,username", "access_token": token},
        timeout=HTTP_TIMEOUT,
    )
    if not response.ok:
        raise RuntimeError(f"Error al autodescubrir ID de Threads: {response.text}")

    data = response.json()
    return data["id"]


def publicar_en_threads(texto: str, image_url: str | None = None) -> str:
    texto = texto.strip()
    if not texto:
        raise ValueError("El texto del post está vacío.")
    if len(texto) > MAX_POST_CHARS:
        raise ValueError(f"El texto supera el límite de {MAX_POST_CHARS} caracteres de Threads.")

    token = env_required("THREADS_ACCESS_TOKEN")
    user_id = obtener_threads_user_id(token)

    # Paso 1: Crear el contenedor de medios en Threads
    container_url = f"{THREADS_API_BASE}/{user_id}/threads"
    payload = {
        "media_type": "TEXT" if not image_url else "IMAGE",
        "text": texto,
        "access_token": token,
    }
    if image_url:
        payload["image_url"] = image_url

    res_container = requests.post(container_url, data=payload, timeout=HTTP_TIMEOUT)
    if not res_container.ok:
        raise RuntimeError(
            f"Threads rechazó el contenedor (HTTP {res_container.status_code}): {res_container.text}"
        )

    creation_id = res_container.json().get("id")
    if not creation_id:
        raise RuntimeError("No se obtuvo creation_id de Threads.")

# =========================================================
# PAUSA Y VERIFICACION DE ESTADO
# =========================================================
    time.sleep(3) 

    for _ in range(5):
        status_res = requests.get(
            f"{THREADS_API_BASE}/{creation_id}",
            params={"fields": "status", "access_token": token},
            timeout=HTTP_TIMEOUT,
        )
        if status_res.ok:
            status = status_res.json().get("status")
            if status == "FINISHED":
                break
            elif status in ("ERROR", "EXPIRED"):
                raise RuntimeError(
                    f"El contenedor de Threads falló en Meta con estado: {status}"
                )
        time.sleep(2)
    
    # Paso 2: Publicar el contenedor creado
    publish_url = f"{THREADS_API_BASE}/{user_id}/threads_publish"
    pub_payload = {
        "creation_id": creation_id,
        "access_token": token,
    }

    res_publish = requests.post(publish_url, data=pub_payload, timeout=HTTP_TIMEOUT)
    if not res_publish.ok:
        raise RuntimeError(
            f"Threads rechazó la publicación (HTTP {res_publish.status_code}): {res_publish.text}"
        )

    post_id = res_publish.json().get("id")
    return post_id or creation_id


# =========================================================
# 6. CREWAI: TRES AGENTES (ADAPTADOS A THREADS)
# =========================================================

def ejecutar_agentes(idea: str) -> str:
    idea = idea.strip()
    if not idea:
        raise ValueError("La idea está vacía.")

    gemini = LLM(
        model=os.getenv("GEMINI_MODEL", "gemini/gemini-2.5-flash"),
        api_key=env_required("GEMINI_API_KEY"),
    )
    openai = LLM(
        model=os.getenv("OPENAI_MODEL", "openai/gpt-4o"),
        api_key=env_required("OPENAI_API_KEY"),
    )

    common = {
        "verbose": False,
        "allow_delegation": False,
        "max_iter": 3,
        "max_execution_time": CREW_TIMEOUT,
    }

    redactor = Agent(
        role="Redactor Ejecutivo para Threads",
        goal="Redactar posts concisos, directos y con alto gancho inicial.",
        backstory=(
            "Líder de tecnología experimentado. Sabes comunicar ideas complejas de forma sintética, "
            "sin clichés y en un tono conversacional óptimo para la red de Threads."
        ),
        llm=gemini,
        **common,
    )
    critico = Agent(
        role="Auditor Crítico de Sintaxis y Longitud",
        goal="Garantizar concisión absoluta y eliminación de paja o clichés.",
        backstory=(
            "Editor sintáctico exigente. Aseguras que el texto mantenga un tono natural y "
            "un máximo estricto de 500 caracteres."
        ),
        llm=openai,
        **common,
    )
    editor = Agent(
        role="Editor Final",
        goal="Entregar la versión final pulida lista para aprobación.",
        backstory=(
            "Conservas la autenticidad del autor y entregas únicamente el contenido directo "
            "sin exceder los 500 caracteres."
        ),
        llm=gemini,
        **common,
    )

    tasks = [
        Task(
            description=(
                "Redacta una publicación para Threads basada en esta idea. "
                "Debe ser concisa, impactante y no exceder los 450 caracteres:\n\n" + idea
            ),
            expected_output="Borrador completo del post.",
            agent=redactor,
        ),
        Task(
            description=(
                "Audita el borrador. Elimina rodeos, clichés o frases artificiales. "
                "Verifica que la lectura sea fluida y breve."
            ),
            expected_output="Críticas accionables.",
            agent=critico,
        ),
        Task(
            description=(
                "Aplica las críticas y devuelve ÚNICAMENTE el texto final, "
                "sin introducciones, comillas ni comentarios."
            ),
            expected_output="Post final para Threads (máximo 500 caracteres).",
            agent=editor,
        ),
    ]

    crew = Crew(
        agents=[redactor, critico, editor],
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff()
    text = str(getattr(result, "raw", result)).strip()

    if not text:
        raise RuntimeError("CrewAI devolvió un resultado vacío.")
    if len(text) > MAX_POST_CHARS:
        raise ValueError(
            f"El borrador supera {MAX_POST_CHARS} caracteres; solicita una versión más corta."
        )
    return text


# =========================================================
# 7. VISTAS PERSISTENTES Y APROBACIÓN
# =========================================================

class ApprovalView(discord.ui.View):
    def __init__(self, draft_id: str):
        super().__init__(timeout=None)
        self.draft_id = draft_id

        approve = discord.ui.Button(
            label="Aprobar y publicar en Threads",
            style=discord.ButtonStyle.success,
            emoji="🚀",
            custom_id=f"threads:approve:{draft_id}",
        )
        approve.callback = self.approve
        self.add_item(approve)

        discard = discord.ui.Button(
            label="Descartar",
            style=discord.ButtonStyle.danger,
            emoji="🗑️",
            custom_id=f"threads:discard:{draft_id}",
        )
        discard.callback = self.discard
        self.add_item(discard)

    async def approve(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)

        if not is_authorized(interaction.user.id):
            await interaction.followup.send("No tienes autorización.", ephemeral=True)
            return

        draft = get_draft(self.draft_id)
        if not draft:
            await interaction.followup.send("No se encontró el borrador.", ephemeral=True)
            return

        if not claim_draft(self.draft_id):
            current = get_draft(self.draft_id)
            state = current["status"] if current else "no disponible"
            await interaction.followup.send(
                f"El borrador ya fue procesado. Estado: {state}.",
                ephemeral=True,
            )
            return

        try:
            loop = asyncio.get_running_loop()
            image_url = None
            if draft["image"]:
                render_url = os.getenv(
                    "RENDER_EXTERNAL_URL","https://threads-agent-bot.onrender.com"
                ).rstrip("/")
                image_url = f"{render_url}/image/{self.draft_id}"
                
            post_id = await loop.run_in_executor(
                executor, publicar_en_threads, draft["text"], None
            )

            change_status(self.draft_id, "PUBLISHED", post_id=post_id)
            self.disable_all_items()

            await interaction.followup.send(
                f"🚀 **¡Publicado exitosamente en Threads!**\nID: `{post_id}`",
                ephemeral=False,
            )
            try:
                await interaction.message.edit(view=self)
            except Exception:
                pass

        except Exception as e:
            change_status(self.draft_id, "PENDING", error=str(e))
            log.exception("Falló la publicación en Threads.")
            await interaction.followup.send(
                f"❌ Error al publicar en Threads: `{str(e)}`",
                ephemeral=True,
            )

    async def discard(self, interaction: discord.Interaction):
        if not is_authorized(interaction.user.id):
            await interaction.response.send_message(
                "No tienes autorización para descartar.", ephemeral=True
            )
            return

        try:
            if not discard_draft(self.draft_id):
                draft = get_draft(self.draft_id)
                state = draft["status"] if draft else "no disponible"
                await interaction.response.send_message(
                    f"No se pudo descartar. Estado actual: {state}.",
                    ephemeral=True,
                )
                return

            self.disable_all_items()
            await interaction.response.edit_message(
                content="Borrador descartado.", view=self
            )
        except Exception:
            log.exception("No se pudo descartar el borrador.")
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    "Error al descartar el borrador.", ephemeral=True
                )

    def disable_all_items(self):
        for item in self.children:
            item.disabled = True


def render_draft_message(draft) -> str:
    preview = draft["text"]
    if len(preview) > PREVIEW_CHARS:
        preview = preview[:PREVIEW_CHARS] + "\n\n[Vista previa recortada.]"
    return (
        f"**Borrador para Threads**\n\n{preview}\n\n"
        f"ID del borrador: `{draft['id']}`"
    )


async def send_draft_preview(ctx, draft) -> None:
    view = ApprovalView(draft["id"])
    content = render_draft_message(draft)

    if draft["image"]:
        file = discord.File(io.BytesIO(bytes(draft["image"])), filename="preview.png")
        await ctx.send(content=content, file=file, view=view)
    else:
        await ctx.send(content=content, view=view)


# =========================================================
# 8. COMANDOS DISCORD
# =========================================================

@bot.command(name="threads")
@commands.guild_only()
async def comando_post(ctx: commands.Context, *, idea: str = ""):
    if not is_authorized(ctx.author.id):
        await ctx.reply("No tienes autorización para usar este bot.")
        return

    idea = idea.strip()
    if not idea:
        await ctx.reply("Uso: `!post tu idea para Threads`")
        return

    if len(idea) > MAX_IDEA_CHARS:
        await ctx.reply(f"La idea supera el máximo de {MAX_IDEA_CHARS} caracteres.")
        return

    status_message = await ctx.send("⚙️ Generando el borrador con CrewAI...")

    image_bytes = None
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]
        try:
            if attachment.size > MAX_IMAGE_BYTES:
                raise ValueError("La imagen supera el límite de 8 MiB.")
            image_bytes = validate_image(await attachment.read())
        except Exception as exc:
            await status_message.edit(content=f"No se pudo usar la imagen: {exc}")
            return

    loop = asyncio.get_running_loop()
    try:
        post_text = await asyncio.wait_for(
            asyncio.shield(loop.run_in_executor(executor, ejecutar_agentes, idea)),
            timeout=CREW_TIMEOUT + 15,
        )
    except asyncio.TimeoutError:
        await status_message.edit(
            content="La generación tardó demasiado. Inténtalo de nuevo más tarde."
        )
        return
    except Exception:
        log.exception("Falló la generación del post.")
        await status_message.edit(
            content="No se pudo generar el borrador. Revisa los logs privados."
        )
        return

    if image_bytes is None:
        try:
            image_bytes = await asyncio.wait_for(
                asyncio.shield(loop.run_in_executor(executor, generar_imagen, post_text)),
                timeout=25,
            )
        except Exception:
            log.warning("No se pudo generar imagen; se continuará sin ella.", exc_info=True)
            image_bytes = None

    draft_id = uuid.uuid4().hex
    try:
        save_draft(draft_id, ctx.author.id, post_text, image_bytes)
        draft = get_draft(draft_id)
    except Exception:
        log.exception("No se pudo guardar el borrador.")
        await status_message.edit(content="No se pudo guardar el borrador; no se publicará.")
        return

    try:
        await status_message.delete()
    except discord.HTTPException:
        pass

    try:
        await send_draft_preview(ctx, draft)
    except discord.HTTPException:
        log.exception("No se pudo enviar la vista previa.")
        await ctx.send(
            "No se pudo mostrar la vista previa. El borrador quedó guardado. "
            f"Usa `!recuperar {draft_id}` para volver a mostrarlo."
        )


@bot.command(name="pendientes")
@commands.guild_only()
async def comando_pendientes(ctx: commands.Context):
    if not is_authorized(ctx.author.id):
        await ctx.reply("No tienes autorización para consultar borradores.")
        return

    try:
        drafts = list_drafts("PENDING", limit=15)
    except Exception:
        log.exception("No se pudieron consultar borradores pendientes.")
        await ctx.reply("No se pudo consultar la base de datos.")
        return

    if not drafts:
        await ctx.reply("No hay borradores pendientes.")
        return

    lines = ["**Borradores pendientes**"]
    for draft in drafts:
        short_text = draft["text"].replace("\n", " ")
        if len(short_text) > 100:
            short_text = short_text[:100] + "…"
        lines.append(f"• `{draft['id']}` — {short_text}")

    lines.append("\nUsa `!recuperar ID` para volver a mostrar uno.")
    await ctx.reply("\n".join(lines))


@bot.command(name="recuperar")
@commands.guild_only()
async def comando_recuperar(ctx: commands.Context, draft_id: str = ""):
    if not is_authorized(ctx.author.id):
        await ctx.reply("No tienes autorización para recuperar borradores.")
        return
    if not re.fullmatch(r"[0-9a-f]{32}", draft_id or ""):
        await ctx.reply("Uso: `!recuperar ID_DEL_BORRADOR`")
        return

    try:
        draft = get_draft(draft_id)
    except Exception:
        log.exception("No se pudo recuperar el borrador.")
        await ctx.reply("No se pudo consultar la base de datos.")
        return

    if not draft:
        await ctx.reply("No se encontró el borrador.")
        return
    if draft["status"] != "PENDING":
        await ctx.reply(f"El borrador no está pendiente. Estado: `{draft['status']}`.")
        return

    try:
        await send_draft_preview(ctx, draft)
    except discord.HTTPException:
        log.exception("No se pudo volver a mostrar el borrador.")
        await ctx.reply("Discord no pudo mostrar el borrador.")


@bot.command(name="estado")
@commands.guild_only()
async def comando_estado(ctx: commands.Context, draft_id: str = ""):
    if not is_authorized(ctx.author.id):
        await ctx.reply("No tienes autorización para consultar estados.")
        return
    if not re.fullmatch(r"[0-9a-f]{32}", draft_id or ""):
        await ctx.reply("Uso: `!estado ID_DEL_BORRADOR`")
        return

    try:
        draft = get_draft(draft_id)
    except Exception:
        log.exception("No se pudo consultar el estado.")
        await ctx.reply("No se pudo consultar la base de datos.")
        return

    if not draft:
        await ctx.reply("No se encontró el borrador.")
        return

    text = (
        f"ID: `{draft['id']}`\n"
        f"Estado: `{draft['status']}`\n"
        f"Creado: `{draft['created_at']}`\n"
        f"ID Threads: `{draft['post_id'] or 'no disponible'}`"
    )
    if draft["error"]:
        text += f"\nÚltimo error: {draft['error'][:500]}"
    await ctx.reply(text)


@bot.event
async def on_ready():
    log.info("Bot conectado como %s", bot.user)


# =========================================================
# 9. ARRANQUE
# =========================================================

if __name__ == "__main__":
    check_configuration()
    init_db()

    flask_thread = threading.Thread(
        target=run_flask,
        daemon=True,
        name="flask-health"
    )
    flask_thread.start()

    import time
    time.sleep(1)

    try:
        bot.run(env_required("DISCORD_BOT_TOKEN"), log_handler=None)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

