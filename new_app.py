import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models
diabetes_model = pickle.load(open(f'{working_dir}Disease prediction and classification\saved_models\diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Health Assistant',
                           ['Chat', 'Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
                           menu_icon='hospital-fill',
                           icons=['chat', 'activity', 'heart', 'person'],
                           default_index=0)

# Chat Page
if selected == 'Chat':
    import streamlit as st
    from streamlit_chat import message
    from langchain.chains import ConversationalRetrievalChain
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.llms import CTransformers
    from langchain.llms import Replicate
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores import FAISS
    from langchain.memory import ConversationBufferMemory
    from langchain.document_loaders import PyPDFLoader
    from langchain.document_loaders import TextLoader
    from langchain.document_loaders import Docx2txtLoader
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    import os
    from dotenv import load_dotenv
    import tempfile


    load_dotenv()


    def initialize_session_state():
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey! 👋"]

    def conversation_chat(query, chain, history):
        result = chain({"question": query, "chat_history": history})
        history.append((query, result["answer"]))
        return result["answer"]

    def display_chat_history(chain):
        reply_container = st.container()
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                with st.spinner('Generating response...'):
                    output = conversation_chat(user_input, chain, st.session_state['history'])

                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with reply_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

    def create_conversational_chain(vector_store):
        load_dotenv()
        # Create llm
        # llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        #                     streaming=True, 
        #                     callbacks=[StreamingStdOutCallbackHandler()],
        #                     model_type="llama", config={'max_new_tokens': 2048, 'temperature': 0.01, 'context_length': 4096})
        llm = Replicate(
            streaming = True,
            model = "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781", 
            callbacks=[StreamingStdOutCallbackHandler()],
            input = {"temperature": 0.01, "max_length" :4500,"top_p":1})
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                    memory=memory)
        return chain

    def main():
        load_dotenv()
        # Initialize session state
        initialize_session_state()
        st.title("MediAssist 👨‍⚕️🚑🩺👨‍💻 :")
        # Initialize Streamlit
        st.sidebar.title("Knowledge Processing")
        uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)


        if uploaded_files:
            text = []
            for file in uploaded_files:
                file_extension = os.path.splitext(file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file.read())
                    temp_file_path = temp_file.name

                loader = None
                if file_extension == ".pdf":
                    loader = PyPDFLoader(temp_file_path)
                elif file_extension == ".docx" or file_extension == ".doc":
                    loader = Docx2txtLoader(temp_file_path)
                elif file_extension == ".txt":
                    loader = TextLoader(temp_file_path)

                if loader:
                    text.extend(loader.load())
                    os.remove(temp_file_path)

            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
            text_chunks = text_splitter.split_documents(text)

            # Create embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                            model_kwargs={'device': 'cpu'})

            # Create vector store
            vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

            # Create the chain object
            chain = create_conversational_chain(vector_store)

            
            display_chat_history(chain)

    if __name__ == "__main__":
        main()



# Diabetes Prediction Page
if selected == 'Tests Prediction':
    import os
    import pickle
    import streamlit as st
    from streamlit_option_menu import option_menu

    # Set page configuration
    st.set_page_config(page_title="Health Assistant",
                    layout="wide",
                    page_icon="🧑‍⚕️")

        
    # getting the working directory of the main.py
    working_dir = os.path.dirname(os.path.abspath(__file__))

    # loading the saved models

    diabetes_model = pickle.load(open(f'{working_dir}DPC\saved_models\diabetes_model.sav', 'rb'))

    heart_disease_model = pickle.load(open(f'{working_dir}DPC\saved_models\heart_disease_model.sav', 'rb'))

    parkinsons_model = pickle.load(open(f'{working_dir}DPC\saved_models\parkinsons_model.sav', 'rb'))

    # sidebar for navigation
    with st.sidebar:
        selected = option_menu('Multiple Disease Prediction System',

                            ['Diabetes Prediction',
                                'Heart Disease Prediction',
                                'Parkinsons Prediction'],
                            menu_icon='hospital-fill',
                            icons=['activity', 'heart', 'person'],
                            default_index=0)


    # Diabetes Prediction Page
    if selected == 'Diabetes Prediction':

        # page title
        st.title('Diabetes Prediction using ML')

        # getting the input data from the user
        col1, col2, col3 = st.columns(3)

        with col1:
            Pregnancies = st.text_input('Number of Pregnancies')

        with col2:
            Glucose = st.text_input('Glucose Level')

        with col3:
            BloodPressure = st.text_input('Blood Pressure value')

        with col1:
            SkinThickness = st.text_input('Skin Thickness value')

        with col2:
            Insulin = st.text_input('Insulin Level')

        with col3:
            BMI = st.text_input('BMI value')

        with col1:
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

        with col2:
            Age = st.text_input('Age of the Person')


        # code for Prediction
        diab_diagnosis = ''

        # creating a button for Prediction

        if st.button('Diabetes Test Result'):

            user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                        BMI, DiabetesPedigreeFunction, Age]

            user_input = [float(x) for x in user_input]

            diab_prediction = diabetes_model.predict([user_input])

            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'

        st.success(diab_diagnosis)

    # Heart Disease Prediction Page
    if selected == 'Heart Disease Prediction':

        # page title
        st.title('Heart Disease Prediction using ML')

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.text_input('Age')

        with col2:
            sex = st.text_input('Sex')

        with col3:
            cp = st.text_input('Chest Pain types')

        with col1:
            trestbps = st.text_input('Resting Blood Pressure')

        with col2:
            chol = st.text_input('Serum Cholestoral in mg/dl')

        with col3:
            fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

        with col1:
            restecg = st.text_input('Resting Electrocardiographic results')

        with col2:
            thalach = st.text_input('Maximum Heart Rate achieved')

        with col3:
            exang = st.text_input('Exercise Induced Angina')

        with col1:
            oldpeak = st.text_input('ST depression induced by exercise')

        with col2:
            slope = st.text_input('Slope of the peak exercise ST segment')

        with col3:
            ca = st.text_input('Major vessels colored by flourosopy')

        with col1:
            thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

        # code for Prediction
        heart_diagnosis = ''

        # creating a button for Prediction

        if st.button('Heart Disease Test Result'):

            user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

            user_input = [float(x) for x in user_input]

            heart_prediction = heart_disease_model.predict([user_input])

            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person is having heart disease'
            else:
                heart_diagnosis = 'The person does not have any heart disease'

        st.success(heart_diagnosis)

    # Parkinson's Prediction Page
    if selected == "Parkinsons Prediction":

        # page title
        st.title("Parkinson's Disease Prediction using ML")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            fo = st.text_input('MDVP:Fo(Hz)')

        with col2:
            fhi = st.text_input('MDVP:Fhi(Hz)')

        with col3:
            flo = st.text_input('MDVP:Flo(Hz)')

        with col4:
            Jitter_percent = st.text_input('MDVP:Jitter(%)')

        with col5:
            Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

        with col1:
            RAP = st.text_input('MDVP:RAP')

        with col2:
            PPQ = st.text_input('MDVP:PPQ')

        with col3:
            DDP = st.text_input('Jitter:DDP')

        with col4:
            Shimmer = st.text_input('MDVP:Shimmer')

        with col5:
            Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

        with col1:
            APQ3 = st.text_input('Shimmer:APQ3')

        with col2:
            APQ5 = st.text_input('Shimmer:APQ5')

        with col3:
            APQ = st.text_input('MDVP:APQ')

        with col4:
            DDA = st.text_input('Shimmer:DDA')

        with col5:
            NHR = st.text_input('NHR')

        with col1:
            HNR = st.text_input('HNR')

        with col2:
            RPDE = st.text_input('RPDE')

        with col3:
            DFA = st.text_input('DFA')

        with col4:
            spread1 = st.text_input('spread1')

        with col5:
            spread2 = st.text_input('spread2')

        with col1:
            D2 = st.text_input('D2')

        with col2:
            PPE = st.text_input('PPE')

        # code for Prediction
        parkinsons_diagnosis = ''

        # creating a button for Prediction    
        if st.button("Parkinson's Test Result"):

            user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                        RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                        APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

            user_input = [float(x) for x in user_input]

            parkinsons_prediction = parkinsons_model.predict([user_input])

            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"

        st.success(parkinsons_diagnosis)

    # Add your Parkinson's disease prediction functionality here...
