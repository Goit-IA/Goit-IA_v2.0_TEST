# --- web_scraper_vectordb.py ---

import requests
from bs4 import BeautifulSoup
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# --- CONSTANTES DE CONFIGURACIÓN ---

# ¡IMPORTANTE! Aquí debes poner la lista de URLs que quieres procesar.
URLS_A_ESCANEAR = [
    "https://www.uv.mx/estudiantes/tramites-escolares/",
    "https://www.uv.mx/estudiantes/tramites-escolares/tramites-escolares-total/",
    "https://www.uv.mx/estudiantes/tramites-escolares/inscripcion-academico-administrativa/",
    "https://www.uv.mx/estudiantes/tramites-escolares/credencial-estudiante-fisica/",
    "https://www.uv.mx/estudiantes/tramites-escolares/credencial-estudiante-digital/",
    "https://www.uv.mx/estudiantes/tramites-escolares/seguro-facultativo/",
    "https://www.uv.mx/estudiantes/tramites-escolares/examen-de-salud-integral-esi/",
    "https://www.uv.mx/estudiantes/tramites-escolares/cambio-programa-educativo/",
    "https://www.uv.mx/estudiantes/tramites-escolares/reinscripcion-inscripcion-en-linea-il/",
    "https://www.uv.mx/estudiantes/tramites-escolares/declaracion-de-equivalencia-o-revalidacion-de-estudios/",
    "https://www.uv.mx/estudiantes/tramites-escolares/baja-temporal-por-periodo-escolar/",
    "https://www.uv.mx/estudiantes/tramites-escolares/baja-temporal-por-experiencia-educativa/",
    "https://www.uv.mx/estudiantes/tramites-escolares/baja-temporal-extemporanea/",
    "https://www.uv.mx/estudiantes/tramites-escolares/baja-definitiva/",
    "https://www.uv.mx/estudiantes/tramites-escolares/cambio-tutor-academico/",
    "https://www.uv.mx/estudiantes/tramites-escolares/traslado-escolar/",
    "https://www.uv.mx/estudiantes/tramites-escolares/dictamen-para-la-acreditacion-del-idioma-ingles-o-acreditacion-de-la-lengua/",
    "https://www.uv.mx/estudiantes/tramites-escolares/transferencia-de-calificacion-de-ee-a-otro-programa-educativo/",
    "https://www.uv.mx/estudiantes/tramites-escolares/equivalencia-y-o-transferencia-de-calificacion-de-lengua-i-ii-o-ingles-i-ii-al-mismo-programa-educativo-que-cursa-el-alumno/",
    "https://www.uv.mx/estudiantes/tramites-escolares/reconocimiento-de-creditos-por-ee-acreditadas-en-programas-educativos-cursados-previamente/",
    "https://www.uv.mx/estudiantes/tramites-escolares/transferencia-de-creditos-para-el-afel-a-traves-de-las-ee-de-centros-de-idiomas/,"
    "https://www.uv.mx/estudiantes/tramites-escolares/movilidad-estudiantil-institucional/",
    "https://www.uv.mx/estudiantes/tramites-escolares/movilidad-estudiantil-nacional/",
    "https://www.uv.mx/estudiantes/tramites-escolares/movilidad-estudiantil-internacional/",
    "https://www.uv.mx/estudiantes/tramites-escolares/cumplimiento-de-servicio-social/",
    "https://www.uv.mx/estudiantes/tramites-escolares/acreditacion-de-la-experiencia-recepcional/",
    "https://www.uv.mx/estudiantes/tramites-escolares/certificado-de-estudios-completo-o-incompleto/",
    "https://www.uv.mx/estudiantes/tramites-escolares/legalizacion-de-certificados-de-estudio/",
    "https://www.uv.mx/estudiantes/tramites-escolares/expedicion-de-titulo-diploma-y-grado-academico/",
    "https://www.uv.mx/estudiantes/tramites-escolares/cedula-profesional/",
    "http://subsegob.veracruz.gob.mx/documentos.php",
    "https://www.uv.mx/estudiantes/tramites-escolares/registro-de-inicio-y-liberacion-del-servicio-social/",
    "https://www.uv.mx/estudiantes/tramites-escolares/autorizacion-de-examen-profesional-o-exencion/",
    "https://www.uv.mx/estudiantes/tramites-escolares/expedicion-de-carta-de-pasante/",
    "https://www.uv.mx/estudiantes/tramites-escolares/autorizacion-de-examen-de-grado/",
    "https://www.uv.mx/dgrf/files/2025/02/Tabulador-de-Cuotas-por-Serv.-Acad.-y-Admvos.-UV-febrero-2025.pdf",
    "https://www.uv.mx/dgae/circulares/",
    "https://www.uv.mx/secretariaacademica/cuotas-del-comite-pro-mejoras/",
    "https://www.uv.mx/legislacion/files/2023/01/RComite%CC%81sProMejoras2023.pdf",
    "https://www.uv.mx/transparencia/ot875/comite-pro-mejoras/",
    "https://www.uv.mx/orizaba/negocios/informes-de-situacion-financiera/",
    "https://www.uv.mx/secretariaacademica/files/2024/09/CIRCULAR-005-2023.pdf",
    "https://www.uv.mx/secretariaacademica/files/2024/11/lineamientos-cuotas-2025.pdf"
]

# Nombre de la carpeta donde se guardará la base de datos vectorial
CHROMA_PATH = "chroma_db_web" 

# Modelo de embeddings que usará Ollama. "nomic-embed-text" es una excelente opción.
MODELO_EMBEDDING = "nomic-embed-text" 

def raspar_y_limpiar_urls(urls):
    """
    Recopila el contenido de una lista de URLs, lo limpia y lo convierte
    en una lista de objetos Document de LangChain.
    """
    print("Iniciando el proceso de web scraping...")
    documentos_procesados = []

    for url in urls:
        try:
            print(f"Procesando: {url}")
            
            # 1. Realizar la petición HTTP para obtener el HTML
            headers = {'User-Agent': 'Mozilla/5.0'} # Simular un navegador
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status() # Lanza un error si la petición falla

            # 2. Analizar el HTML con Beautiful Soup
            soup = BeautifulSoup(response.content, 'html.parser')

            # 3. Eliminar etiquetas innecesarias (navegación, scripts, etc.)
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()

            # 4. Extraer el texto y limpiarlo
            texto_crudo = soup.get_text()
            
            # Limpieza con expresiones regulares (regex) para eliminar espacios extra
            lineas = (line.strip() for line in texto_crudo.splitlines())
            fragmentos = (frase.strip() for line in lineas for frase in line.split("  "))
            texto_limpio = '\n'.join(f for f in fragmentos if f)

            if texto_limpio:
                # 5. Crear un objeto Document para LangChain
                # Se guarda la URL en los metadatos para saber la fuente.
                nuevo_documento = Document(page_content=texto_limpio, metadata={"source": url})
                documentos_procesados.append(nuevo_documento)
            
        except requests.RequestException as e:
            print(f"Error al intentar acceder a la URL {url}: {e}")

    print(f"Se procesaron exitosamente {len(documentos_procesados)} páginas web.")
    return documentos_procesados

def dividir_documentos(documentos):
    """
    Divide los documentos en fragmentos (chunks) más pequeños.
    Esta función es idéntica a la de tu script original.
    """
    if not documentos:
        return None
        
    print("Dividiendo documentos en fragmentos (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len,
        add_start_index=True # Ayuda a identificar el origen del fragmento
    )
    fragmentos = text_splitter.split_documents(documentos)
    print(f"Los documentos se dividieron en {len(fragmentos)} fragmentos.")
    return fragmentos

def crear_y_guardar_vectordb(fragmentos):
    """
    Crea y guarda una base de datos vectorial ChromaDB a partir de los fragmentos.
    Esta función es idéntica a la de tu script original.
    """
    if not fragmentos:
        print("No hay fragmentos para procesar.")
        return

    print("Creando la base de datos vectorial con ChromaDB...")
    
    # Inicializa el modelo de embeddings de Ollama
    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING)
    
    # Crea y guarda la base de datos en un solo paso
    vectorstore = Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print(f"¡Base de datos guardada exitosamente en la carpeta '{CHROMA_PATH}'!")

# --- BLOQUE DE EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    # Paso 1: Obtener y limpiar el contenido de las páginas web
    documentos_web = raspar_y_limpiar_urls(URLS_A_ESCANEAR)
    
    if documentos_web:
        # Paso 2: Dividir los documentos en fragmentos
        fragmentos_de_texto = dividir_documentos(documentos_web)
        
        # Paso 3: Crear la base de datos vectorial
        if fragmentos_de_texto:
            crear_y_guardar_vectordb(fragmentos_de_texto)
            
    print("\nProceso completado. Tu base de datos vectorial está lista para ser usada.")