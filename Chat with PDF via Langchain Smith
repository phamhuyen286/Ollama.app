# %%
import os
import warnings
from dotenv import load_dotenv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

load_dotenv()

# %%
os.environ['LANGCHAIN_PROJECT']

# %% [markdown]
# ![Document Ingestion.png](<attachment:Document Ingestion.png>)

# %% [markdown]
# This is only used for Text pdf reader, for OCR scan pdf we will be using different which is PDFOCRLoader

# %%
#import PyMuPDF refer https://python.langchain.com/docs/integrations/document_loaders/pymupdf/

from langchain_community.document_loaders import PyMuPDFLoader

file_path = "./rag-dataset/dietary supplements.pdf"
loader = PyMuPDFLoader(file_path)

docs = loader.load()

doc = docs[0]


# %%
# import all documents in the folder

import os
pdfs = []
for roots, dirs, files in os.walk('rag-dataset'):
    #print(roots, dirs, files)
    for file in files:
        if file.endswith('.pdf'):
            pdfs.append(os.path.join(roots, file))

# %%
pdfs

# %%
docs = []

for pdf in pdfs:
    loader = PyMuPDFLoader(pdf)
    pages = loader.load()
    docs.extend(pages)

# %%
len(docs)

# %% [markdown]
# Right now, we are going to divided text into documents
# 
# #### Document Chunking
# 
# we are using Recursive Text Splitter from langchain : https://python.langchain.com/docs/how_to/recursive_text_splitter/
# 
# we need to care on context window and overlap window of the following code

# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap =100) # means we are using context is 1000 character as chunks size  and overlap = 100 total characters to be used

chunks = text_splitter.split_documents(docs)

# %%
len(docs), len(chunks) # around each pages we get 311/64 ~20% of documents

# %%
print(chunks[0].page_content)

# %%
len(chunks[0].page_content)

# %%
# to know the total number of token is there in page content in docs and in chunks (resulted 795 and 271)

import tiktoken

encoding = tiktoken.encoding_for_model('gpt-4o-mini')

len(encoding.encode(docs[0].page_content)), len(encoding.encode(chunks[0].page_content))

# %% [markdown]
# Please noted that LLM context should be larger than total token we have in the chunks
# 
# Now we will download embedding model: Ollama nomic text to process further ollama pull nomic-embed-text https://ollama.com/library/nomic-embed-text from the link we also see this model has number of context is 8192

# %% [markdown]
# ### Document Vector Embedding

# %%
from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# %%
embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url='http://localhost:11434')
single_vector = embeddings.embed_query('this is some text')

len(single_vector) # vector size 768 items

# %%
index = faiss.IndexFlatL2(len(single_vector))

index.ntotal, index.d

# %%
vector_store = FAISS(
    embedding_function = embeddings,
    index=index,
    docstore=InMemoryDocstore(), # because we store docs here in the memory only
    index_to_docstore_id= {} #keep it empty to load vector later on
)

# %%
vector_store

# %%
#add chunks into Vector store

ids = vector_store.add_documents(documents = chunks)

#help(vector_store) to understannd about vector store

# %%
vector_store.index_to_docstore_id

# %% [markdown]
# the following code please remember only run once time, once it is run, you keep it as comments to avoid any error while we continue to do

# %%
#store vector database
db_name ='health_supplements' # to store your data
vector_store.save_local(db_name)

#load vector database
new_vector_store = FAISS.load_local(db_name, embeddings=embeddings, allow_dangerous_deserialization=True)

# %% [markdown]
# ![Pasted Graphic 7.png](<attachment:Pasted Graphic 7.png>)

# %% [markdown]
# ### Retreival

# %%
question = 'what is used to gain muscle mass? '

docs = vector_store.search(query=question, search_type='similarity')

for doc in docs:
    print(doc.page_content)
    print('\n\n')

# %%
retriever = vector_store.as_retriever(search_type='mmr', search_kwargs = {'k':3, 
                                                                          'fetch_k':100, 
                                                                          'lambda_mult': 1}) # return relevant input into the answer
#in search_type you can change according to your data type

# %%
retriever.invoke(question)

# for doc in docs:
#     print(doc.page_content)
#     print('\n\n')



# %%
question = 'what are the benefits of BCAA supplements'
docs = retriever.invoke(question)

# %%
question = 'what are the benefits of supplements'
docs = retriever.invoke(question)

# %%
question = 'what is used to reduced weight'
docs = retriever.invoke(question)

# %% [markdown]
# ### RAG with LLAMA 3.2 on OLLAMA

# %%
#passing all chunks data along with question via LLAMA to get final response

from langchain import hub #to pull out relevant RAG https://smith.langchain.com/hub?organizationId=6a0af89d-44f3-4478-b631-03cc68c3f5a4
from langchain_core.output_parsers import StrOutputParser # to get string data
from langchain_core.runnables import RunnablePassthrough # pass context to LLM
from langchain_core.prompts import ChatPromptTemplate # pass our prompt to chat prompt context so we can pass our question

from langchain_ollama import ChatOllama #make a connection btw Chat prompt Ollama to your langchain framework

# %%
model = ChatOllama(model = 'llama3.2:3b', base_url= 'http://localhost:11434')

model.invoke('hi')

# %%
#method 1 to write prompt
prompt = hub.pull("daethyra/rag-prompt") # this one prompt template from this link: https://smith.langchain.com/hub/daethyra/rag-prompt?organizationId=6a0af89d-44f3-4478-b631-03cc68c3f5a4 but you can choose different type as you like


# %%
prompt

# %%
#or method 2 you can create your own prompt

prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only.
    Question: {question} 
    Context: {context} 
    Answer:

"""

prompt = ChatPromptTemplate.from_template(prompt)

# %%
prompt

# %%
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

#print(format_docs(docs))

# %%
rag_chain = (
    {"context": retriever|format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# %%
#question = 'what is used to reduce weight'
#question = 'what is used to gain muscle mass?'
#question = 'what is side effects of supplement?'
question = 'what is used to increase mass of the Earth?'

output = rag_chain.invoke(question)
print(output)


# %%
#Please give credit to KGP Talkie Youtube channel to make the concept clearly step-wise

# %%



