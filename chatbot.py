import sys
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- CONSTANTES DE CONFIGURACIÓN ---
CHROMA_PATH = "chroma_db"
MODELO_OLLAMA = "phi3:mini"

def get_rag_chain():
    """
    Configura y devuelve la cadena de RAG completa.
    """
    try:
        # Cargar la base de datos vectorial desde el disco
        embeddings = OllamaEmbeddings(model=MODELO_OLLAMA)
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embeddings
        )
        
        # Configurar el retriever
        retriever = vectorstore.as_retriever()

        # Crear la plantilla del prompt
        template = """
        Responde a la siguiente pregunta basándote únicamente en el contexto proporcionado.
        Si la información no está en el contexto, responde: "No tengo información suficiente sobre eso en mis documentos."

        Contexto:
        {context}

        Pregunta: 
        {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # Inicializar el modelo de lenguaje
        llm = Ollama(model=MODELO_OLLAMA)

        # Ensamblar la cadena de RAG
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain

    except FileNotFoundError:
        print(f"Error: La base de datos en '{CHROMA_PATH}' no fue encontrada.")
        print("Asegúrate de haber ejecutado primero el script 'preparar_vectordb.py'.")
        sys.exit(1) # Termina el programa si no encuentra la base de datos
    except Exception as e:
        print(f"Ocurrió un error inesperado al cargar la cadena RAG: {e}")
        sys.exit(1)


def main():
    """
    Función principal que inicia el chatbot interactivo.
    """
    print("🤖 Iniciando chatbot... (Esto puede tardar un momento)")
    
    # Cargar la cadena de RAG una sola vez al inicio
    rag_chain = get_rag_chain()
    
    print("\n✅ Chatbot listo. Escribe tu pregunta o 'salir' para terminar.")
    print("-" * 60)

    while True:
        # Pedir input al usuario
        pregunta = input("Tú: ")
        
        # Salir del bucle si el usuario escribe 'salir'
        if pregunta.lower() == 'salir':
            print("\n🤖 ¡Hasta luego!")
            break
        
        # Invocar la cadena y mostrar la respuesta
        try:
            respuesta = rag_chain.invoke(pregunta)
            print(f"Chatbot: {respuesta}")
        except Exception as e:
            print(f"\nOcurrió un error al procesar la pregunta: {e}")

# --- PUNTO DE ENTRADA PRINCIPAL ---
if __name__ == "__main__":
    main()