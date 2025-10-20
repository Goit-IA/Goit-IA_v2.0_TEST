import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings # Importación actualizada

# --- CONSTANTES DE CONFIGURACIÓN ---
DATA_PATH = "data/"
CHROMA_PATH = "chroma_db"
MODELO_OLLAMA = "phi3:mini"

def cargar_documentos():
    """
    Carga todos los documentos soportados desde la carpeta 'data'.
    """
    print("Cargando documentos desde la carpeta 'data'...")
    loader_pdf = DirectoryLoader(
        DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, 
        show_progress=True, use_multithreading=True
    )
    loader_txt = DirectoryLoader(
        DATA_PATH, glob="**/*.txt", loader_cls=TextLoader, 
        show_progress=True, use_multithreading=True
    )
    
    documentos = loader_pdf.load() + loader_txt.load()
    
    if not documentos:
        print("No se encontraron documentos en la carpeta 'data'.")
        return None
        
    print(f"Se cargaron {len(documentos)} páginas/documentos.")
    return documentos

def dividir_documentos(documentos):
    """
    Divide los documentos en fragmentos (chunks) más pequeños.
    """
    if not documentos:
        return None
        
    print("Dividiendo documentos en fragmentos (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    fragmentos = text_splitter.split_documents(documentos)
    print(f"Los documentos se dividieron en {len(fragmentos)} fragmentos.")
    return fragmentos

def crear_y_guardar_vectordb(fragmentos):
    """
    Crea y guarda una base de datos vectorial ChromaDB a partir de los fragmentos.
    """
    if not fragmentos:
        print("No hay fragmentos para procesar.")
        return

    print("Creando la base de datos vectorial con ChromaDB...")
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Crear y persistir la base de datos en un solo paso
    vectorstore = Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    print(f"¡Base de datos guardada exitosamente en la carpeta '{CHROMA_PATH}'!")

if __name__ == "__main__":
    # --- BLOQUE DE EJECUCIÓN PRINCIPAL ---
    # Este bloque está comentado porque la base de datos ya fue creada.
    # Para regenerarla (ej. si agregas nuevos archivos en 'data'):
    # 1. Borra la carpeta 'chroma_db'.
    # 2. Descomenta las siguientes líneas.
    # 3. Vuelve a ejecutar el script.
    
    # documentos_cargados = cargar_documentos()
    # fragmentos_de_texto = dividir_documentos(documentos_cargados)
    # if fragmentos_de_texto:
    #     crear_y_guardar_vectordb(fragmentos_de_texto)

    print("Script 'preparar_vectordb.py' está completo y actualizado.")
    print("La base de datos ya existe. Podemos proceder a la Fase 4.")
