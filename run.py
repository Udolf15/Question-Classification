from modelCreation import *
from embeddings import *
from preprocessing import *

preProcessor = Preprocessing()
batch_size = 32
SEQ_LEN = 80

modelObject = ModelC()
embeddings = Embeddings()

train_df = preProcessor.read_csv('data/questions.csv')
train_df = preProcessor.dfPreprocess(train_df)

emb = embeddings.create_embeddings()

stringStop = "के का एक में की है यह और से हैं को पर इस होता कि जो कर मे गया करने किया लिये अपने ने बनी नहीं तो ही या एवं दिया हो इसका था द्वारा हुआ तक साथ करना वाले बाद लिए आप कुछ सकते किसी ये इसके सबसे इसमें थे दो होने वह वे करते बहुत कहा वर्ग कई करें होती अपनी उनके थी यदि हुई जा ना इसे कहते जब होते कोई हुए व न अभी जैसे सभी करता उनकी तरह उस आदि कुल एस रहा इसकी सकता रहे उनका इसी रखें अपना पे उसके \" \' [ ] . , ! : ; ?"
QuestionsPre = ['का मोदी सरकार भारत की संवैधािनक संसाओ','पाथेर पाचाली फिल्म को विदेशी पुरस्कार से कब सम्मानित किया गया','इस बार कौन सी फिल्म हिट होगी ','राष्ट्रीय स्वयंसेवक संघ का मुख्यालय कहाँ है ?','टाइगर श्रॉफ की पहली फिल्म का क्या नाम है','फिल्म का क्या नाम है']

stopWords = preProcessor.StopWords(stringStop)

choice = int(input("Press 1: Trained Model\nPress 2: Train a new Model : "))

if(choice == 2):
    model = modelObject.create_model()
    dg = preProcessor.batch_gen(train_df, batch_size, SEQ_LEN, stopWords, emb)
    model = modelObject.train(model, dg, train_df, batch_size, stopWords, emb, SEQ_LEN, preProcessor)

    choice2 = int(input("Press 1: Save Model\nPress 2: Skip saving : "))
    
    if(choice2 == 1):
        modelObject.save_model(model)
    
elif(choice == 1):
    try:
        mPath = input("Enter the model Path")
        model = load_model(mPath)
    except:
        print("Model not found")

else:
    print("Wrong choice")
    exit

choice = int(input("Press 1: Want to give custom input for test\nPress 2: Use Predefined input for test : "))

if(choice == 1):
    Questions = []
    NoOfQues = int(input("Enter the number of questions : "))

    for _ in range(NoOfQues):
        Questions.append(input("Enter the text in hindi language : "))

    modelObject.predict(model,dg,Questions, stopWords,emb,SEQ_LEN,preProcessor)
else:
    modelObject.predict(model,dg,QuestionsPre, stopWords,emb,SEQ_LEN,preProcessor)