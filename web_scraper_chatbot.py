import sys
import time
import threading

from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- CONSTANTES DE CONFIGURACIÓN ---
CHROMA_PATH = "chroma_db_web"
MODELO_OLLAMA = "phi3:mini"
MODELO_EMBEDDING = "nomic-embed-text"

# --- FUNCIÓN DE CONFIGURACIÓN DE LA CADENA RAG ---
# (CAMBIO) Se eliminó el bloque try/except para que los errores se propaguen hacia arriba.
def get_rag_chain():
    """
    Configura y devuelve la cadena de RAG (Retrieval-Augmented Generation) completa.
    Si ocurre un error durante la configuración (ej. no se encuentra Chroma),
    la excepción será lanzada para que la función que llama la maneje.
    """
    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING)
    
    # Esta es la línea que probablemente podría fallar si la carpeta no existe.
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={'k': 4})

    template ="""
    Actúa como un asistente virtual experto y muy servicial de la Universidad Veracruzana. 
    Tu misión es proporcionar respuestas extremadamente detalladas y completas, utilizando únicamente la información encontrada en el CONTEXTO proporcionado.

    Sigue estas reglas estrictamente:
    1.  **Sé Exhaustivo:** Extrae y sintetiza TODA la información relevante del contexto que responda a la pregunta del usuario. No omitas detalles, requisitos, fechas o pasos mencionados.
    2.  **Organiza la Información:** Estructura tu respuesta de una manera clara y fácil de entender. Si la pregunta es sobre un proceso, descríbelo en una lista ordenada (paso a paso). Si se listan requisitos, usa viñetas.
    3.  **Elabora la Respuesta:** No te limites a extraer texto. Explica los conceptos con tus propias palabras (basadas en el contexto) para que la respuesta sea coherente y completa. El objetivo es que el usuario entienda el tema a fondo.
    4.  **Restricción Absoluta:** Si la información necesaria para responder la pregunta no se encuentra en el CONTEXTO, DEBES responder única y exclusivamente con la frase: "No tengo información suficiente sobre eso en mis documentos." No intentes adivinar ni añadir información externa.

    ---
    CONTEXTO:
    {context}
    ---
    PREGUNTA DEL USUARIO:
    {question}
    ---

    RESPUESTA DETALLADA Y COMPLETA:
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = Ollama(model=MODELO_OLLAMA)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- FUNCIÓN PARA LA ANIMACIÓN DE CARGA --- (Sin cambios)
def mostrar_barra_de_carga(evento_parada):
    """
    Muestra una barra de progreso animada en la consola hasta que
    el 'evento_parada' se active.
    """
    bar_length = 30
    i = 0
    while not evento_parada.is_set():
        progreso = i % 100
        longitud_llena = int(bar_length * progreso // 100)
        barra = '█' * longitud_llena + '-' * (bar_length - longitud_llena)
        sys.stdout.write(f'\rChatbot: Procesando... |{barra}| {progreso}%')
        sys.stdout.flush()
        i += 2
        time.sleep(0.1)

    barra_final = '█' * bar_length
    sys.stdout.write(f'\rChatbot: Procesando... |{barra_final}| 100%\n')
    sys.stdout.flush()

# --- FUNCIÓN PRINCIPAL ---
def main():
    """
    Función principal que inicia el chatbot interactivo.
    """
    print("🤖 Iniciando chatbot con información web...")
    
    # (CAMBIO CLAVE) Se añade un bloque try/except para capturar errores de inicialización.
    try:
        rag_chain = get_rag_chain()
        print("\n✅ Chatbot listo. Pregúntame sobre los trámites de la UV. Escribe 'salir' para terminar.")
        print("-" * 70)
    except Exception as e:
        print("\n❌ ERROR CRÍTICO AL INICIAR EL CHATBOT ❌")
        print(f"No se pudo cargar la cadena de procesamiento de lenguaje. El error fue:\n")
        print(f"   ➡️  {e}\n")
        print("POSIBLES CAUSAS:")
        print(f"   1. La carpeta de la base de datos Chroma ('{CHROMA_PATH}') no existe o está corrupta.")
        print("      Asegúrate de haber ejecutado primero el script que la crea (ej. 'web_scraper_vectordb.py').")
        print(f"   2. El modelo de Ollama ('{MODELO_OLLAMA}' o '{MODELO_EMBEDDING}') no está disponible o no se ha descargado.")
        print("      Verifica que Ollama esté en ejecución y los modelos estén instalados con 'ollama list'.")
        sys.exit(1) # Terminamos el programa de forma controlada porque no puede funcionar.

    while True:
        pregunta = input("Tú: ")
        
        if pregunta.lower() == 'salir':
            print("\n🤖 ¡Hasta luego! Ha sido un placer ayudarte.")
            break
        
        evento_parada = threading.Event()
        resultado = {"respuesta": None, "error": None}

        def obtener_respuesta(p):
            try:
                respuesta = rag_chain.invoke(p)
                resultado["respuesta"] = respuesta
            except Exception as e:
                resultado["error"] = e
            finally:
                evento_parada.set()

        hilo_trabajo = threading.Thread(target=obtener_respuesta, args=(pregunta,))
        hilo_trabajo.start()

        mostrar_barra_de_carga(evento_parada)

        hilo_trabajo.join()

        if resultado["error"]:
            # Este mensaje ahora es para errores DURANTE la conversación.
            print(f"\nLo siento, ocurrió un error al procesar tu pregunta.")
            print(f"Detalle del error: {resultado['error']}")
        else:
            print(f"Chatbot: {resultado['respuesta']}")

if __name__ == "__main__":
    main()