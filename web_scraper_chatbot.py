import sys
import time      # <--- ¡NUEVO! Para controlar la velocidad de la animación
import threading # <--- ¡NUEVO! Para ejecutar la carga en segundo plano

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

def get_rag_chain():
    """
    Configura y devuelve la cadena de RAG (Retrieval-Augmented Generation) completa.
    """
    try:
        embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING)
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

    except Exception as e:
        print(f"Ocurrió un error inesperado al cargar la cadena RAG: {e}")
        print(f"Asegúrate de haber ejecutado primero el script 'web_scraper_vectordb.py' y que la carpeta '{CHROMA_PATH}' exista.")
        sys.exit(1)

# --- FUNCIÓN PARA LA ANIMACIÓN DE CARGA --- (¡NUEVO!)
def mostrar_barra_de_carga(evento_parada):
    """
    Muestra una barra de progreso animada en la consola hasta que
    el 'evento_parada' se active.
    """
    bar_length = 30
    i = 0
    # Mientras el evento no se haya activado (la tarea principal no ha terminado)
    while not evento_parada.is_set():
        # Calculamos un porcentaje que va de 0 a 99 y vuelve a empezar
        # Esto da una sensación de progreso continuo
        progreso = i % 100
        longitud_llena = int(bar_length * progreso // 100)
        barra = '█' * longitud_llena + '-' * (bar_length - longitud_llena)
        
        # El carácter \r al principio mueve el cursor al inicio de la línea
        # para sobreescribirla y crear la animación
        sys.stdout.write(f'\rChatbot: Procesando... |{barra}| {progreso}%')
        sys.stdout.flush()
        
        i += 2
        time.sleep(0.1)

    # Cuando la tarea termina, borramos la barra y mostramos el 100%
    barra_final = '█' * bar_length
    sys.stdout.write(f'\rChatbot: Procesando... |{barra_final}| 100%\n')
    sys.stdout.flush()


def main():
    """
    Función principal que inicia el chatbot interactivo.
    """
    print("🤖 Iniciando chatbot con información web... (Esto puede tardar un momento)")
    rag_chain = get_rag_chain()
    
    print("\n✅ Chatbot listo. Pregúntame sobre los trámites de la UV. Escribe 'salir' para terminar.")
    print("-" * 70)

    while True:
        pregunta = input("Tú: ")
        
        if pregunta.lower() == 'salir':
            print("\n🤖 ¡Hasta luego! Ha sido un placer ayudarte.")
            break
        
        # --- LÓGICA DE CARGA MODIFICADA --- (¡CAMBIO!)

        # 1. Preparar las herramientas para la comunicación entre hilos
        evento_parada = threading.Event()
        resultado = {"respuesta": None, "error": None}

        # 2. Definir la función que hará el trabajo pesado (la llamada al LLM)
        def obtener_respuesta(p):
            try:
                respuesta = rag_chain.invoke(p)
                resultado["respuesta"] = respuesta
            except Exception as e:
                resultado["error"] = e
            finally:
                # 3. Avisar al hilo principal que ya hemos terminado
                evento_parada.set()

        # 4. Crear y empezar el hilo que obtendrá la respuesta
        hilo_trabajo = threading.Thread(target=obtener_respuesta, args=(pregunta,))
        hilo_trabajo.start()

        # 5. Mientras el hilo de trabajo se ejecuta, mostramos la animación en el hilo principal
        mostrar_barra_de_carga(evento_parada)

        # 6. Una vez que la animación termina (porque el hilo de trabajo la detuvo),
        # esperamos a que el hilo termine de limpiarse y mostramos el resultado.
        hilo_trabajo.join()

        if resultado["error"]:
            print(f"\nOcurrió un error al procesar la pregunta: {resultado['error']}")
        else:
            print(f"Chatbot: {resultado['respuesta']}")


if __name__ == "__main__":
    main()