import time
import os
import sys
from finetuning import *

models_folder = '/cluster/work/projects/ec30/fernavr/'


"""
save_path_folder = "./1a-xlm-roberta-base/"
os.makedirs(save_path_folder, exist_ok = True)
with open(save_path_folder+'output_train.txt', 'w') as sys.stdout:
    start = time.time()
    
    train_test(["en"], ["en"], model_path = "/fp/projects01/ec30/models/xlm-roberta-base/", save_path = save_path_folder)
    
    end = time.time()
    print("Time Elapsed: ", end - start)




# ------------------------------------
save_path_folder = "./1a-xlm-roberta-base/"
os.makedirs(save_path_folder, exist_ok = True)
with open(save_path_folder+'output_train.txt', 'w') as sys.stdout:
    start = time.time()
    
    train_test(["en"], ["en"], model_path = "/fp/projects01/ec30/models/xlm-roberta-base/", save_path = save_path_folder)
    
    end = time.time()
    print("Time Elapsed: ", end - start)

# ------------------------------------

save_path_folder = "./1a-bert-base-multilingual-cased/"
os.makedirs(save_path_folder, exist_ok = True)
with open(save_path_folder+'output_train.txt', 'w') as sys.stdout:
    start = time.time()
    train_test(["en"], ["en"], model_path = "/fp/projects01/ec30/models/bert-base-multilingual-cased/", save_path = save_path_folder)
    end = time.time()
    print("Time Elapsed: ", end - start)

# ------------------------------------
"""
save_path_folder = "./1a-electra-base-discriminator/"
os.makedirs(save_path_folder, exist_ok = True)
with open(save_path_folder+'output_train.txt', 'w') as sys.stdout:
    start = time.time()
    train_test(["en"], ["en"], model_path = models_folder+"electra-base-discriminator", save_path = save_path_folder)
    end = time.time()
    print("Time Elapsed: ", end - start)


"""
# ------------------------------------

save_path_folder = "./1a-deberta-base/"
os.makedirs(save_path_folder, exist_ok = True)
with open(save_path_folder+'output_train.txt', 'w') as sys.stdout:
    start = time.time()
    train_test(["en"], ["en"], model_path = "microsoft/deberta-base", save_path = save_path_folder, bs=32)
    end = time.time()
    print("Time Elapsed: ", end - start)



# ------------------------------------

save_path_folder = "./1a-rured2-ner-mdeberta-v3-base/"
os.makedirs(save_path_folder, exist_ok = True)
with open(save_path_folder+'output_train.txt', 'w') as sys.stdout:
    start = time.time()
    train_test(["en"], ["en"], model_path = "Grpp/rured2-ner-mdeberta-v3-base", save_path = save_path_folder, bs=32)
    end = time.time()
    print("Time Elapsed: ", end - start)




# ------------------------------------

save_path_folder = "./1a-mdeberta-v3-base/"
os.makedirs(save_path_folder, exist_ok = True)
with open(save_path_folder+'output_train.txt', 'w') as sys.stdout:
    start = time.time()
    train_test(["en"], ["en"], model_path = "microsoft/mdeberta-v3-base", save_path = save_path_folder, bs=32)
    end = time.time()
    print("Time Elapsed: ", end - start)



# ------------------------------------

save_path_folder = "./1a-deberta-v3-base/"
os.makedirs(save_path_folder, exist_ok = True)
with open(save_path_folder+'output_train.txt', 'w') as sys.stdout:
    start = time.time()
    train_test(["en"], ["en"], model_path = "microsoft/deberta-v3-base", save_path = save_path_folder, bs=32)
    end = time.time()
    print("Time Elapsed: ", end - start)




# ------------------------------------


save_path_folder = "./1b-lr1e-3-mdeberta-v3-base/"
os.makedirs(save_path_folder, exist_ok = True)
with open(save_path_folder+'output_train.txt', 'w') as sys.stdout:
    start = time.time()
    train_test(["en"], ["en"], model_path = "microsoft/mdeberta-v3-base", lr=1e-3, save_path = save_path_folder, bs=32)
    end = time.time()
    print("Time Elapsed: ", end - start)



# ------------------------------------

save_path_folder = "./1b-lr1e-5-mdeberta-v3-base/"
os.makedirs(save_path_folder, exist_ok = True)
with open(save_path_folder+'output_train.txt', 'w') as sys.stdout:
    start = time.time()
    train_test(["en"], ["en"], model_path = "microsoft/mdeberta-v3-base", lr=1e-5, save_path = save_path_folder, bs=32)
    end = time.time()
    print("Time Elapsed: ", end - start)




# ------------------------------------

save_path_folder = "./1b-frozen-mdeberta-v3-base/"
os.makedirs(save_path_folder, exist_ok = True)
with open(save_path_folder+'output_train.txt', 'w') as sys.stdout:
    start = time.time()
    train_test(["en"], ["en"], model_path = "microsoft/mdeberta-v3-base", finetune=False, save_path = save_path_folder, bs=32)
    end = time.time()
    print("Time Elapsed: ", end - start)




# ------------------------------------

# save_path_folder = "./1C-mdeberta-v3-base-ORI/"
# os.makedirs(save_path_folder, exist_ok = True)
# with open(save_path_folder+'output_train.txt', 'w') as sys.stdout:
#     start = time.time()
#     train_test_sep_val(["en"],["en", "it", "af", "sw", "de"], model_path = "microsoft/mdeberta-v3-base",bs=32, save_path = save_path_folder)
#     end = time.time()
#     print("Time Elapsed: ", end - start)


# ------------------------------------


save_path_folder = "./1C-mdeberta-v3-base-SEPARATED/"
os.makedirs(save_path_folder, exist_ok = True)
with open(save_path_folder+'output_train.txt', 'w') as sys.stdout:
    start = time.time()
    train_test_sep_val(["en"],["en", "it", "af", "sw", "de"], model_path = "microsoft/mdeberta-v3-base",bs=32, save_path = save_path_folder)
    end = time.time()
    print("Time Elapsed: ", end - start)



# ------------------------------------


save_path_folder = "./2A-mdeberta-v3-base-SEPARATED/"
os.makedirs(save_path_folder, exist_ok = True)
with open(save_path_folder+'output_train.txt', 'w') as sys.stdout:
    start = time.time()
    train_test_sep_val(["en",  "it"],["en", "it", "af", "sw", "de"], model_path = "microsoft/mdeberta-v3-base",bs=32, save_path = save_path_folder)
    end = time.time()
    print("Time Elapsed: ", end - start)



# ------------------------------------

save_path_folder = "./2B-mdeberta-v3-base-SEPARATED/"
os.makedirs(save_path_folder, exist_ok = True)
with open(save_path_folder+'output_train.txt', 'w') as sys.stdout:
    start = time.time()
    train_test_sep_val(["en",  "de"],["en", "it", "af", "sw", "de"], model_path = "microsoft/mdeberta-v3-base", bs=32, save_path = save_path_folder)
    end = time.time()
    print("Time Elapsed: ", end - start)




# ------------------------------------

save_path_folder = "./2C-mdeberta-v3-base-SEPARATED/"
os.makedirs(save_path_folder, exist_ok = True)
with open(save_path_folder+'output_train.txt', 'w') as sys.stdout:
    start = time.time()
    train_test_sep_val(["en", "it", "de"],["en", "it", "af", "sw", "de"], model_path = "microsoft/mdeberta-v3-base", bs=32, save_path = save_path_folder)
    end = time.time()
    print("Time Elapsed: ", end - start)


# ------------------------------------

save_path_folder = "./2C1-mdeberta-v3-base-SEPARATED/"
os.makedirs(save_path_folder, exist_ok = True)
with open(save_path_folder+'output_train.txt', 'w') as sys.stdout:
    start = time.time()
    train_test_sep_val(["en", "af"],["en", "it", "af", "sw", "de"], model_path = "microsoft/mdeberta-v3-base", bs=32, save_path = save_path_folder)
    end = time.time()
    print("Time Elapsed: ", end - start)




# ------------------------------------

save_path_folder = "./2C2-mdeberta-v3-base-SEPARATED/"
os.makedirs(save_path_folder, exist_ok = True)
with open(save_path_folder+'output_train.txt', 'w') as sys.stdout:
    start = time.time()
    train_test_sep_val(["en", "hu"],["en", "it", "af", "sw", "de"], model_path = "microsoft/mdeberta-v3-base", bs=32, save_path = save_path_folder)
    end = time.time()
    print("Time Elapsed: ", end - start)




# ------------------------------------

save_path_folder = "./2C3-mdeberta-v3-base-SEPARATED/"
os.makedirs(save_path_folder, exist_ok = True)
with open(save_path_folder+'output_train.txt', 'w') as sys.stdout:
    start = time.time()
    train_test_sep_val(["en", "hu", "af"],["en", "it", "af", "sw", "de"], model_path = "microsoft/mdeberta-v3-base", bs=32, save_path = save_path_folder)
    end = time.time()
    print("Time Elapsed: ", end - start)


# ------------------------------------

save_path_folder = "./2C4-mdeberta-v3-base-SEPARATED/"
os.makedirs(save_path_folder, exist_ok = True)
with open(save_path_folder+'output_train.txt', 'w') as sys.stdout:
    start = time.time()
    model = train_test_sep_val(["en", "hu", "af", "it", "de"],["en", "it", "af", "sw", "de"], model_path = "microsoft/mdeberta-v3-base", bs=32, save_path = save_path_folder)
    end = time.time()
    print("Time Elapsed: ", end - start)

# Save the model
model.save_pretrained(save_path_folder)

# Save the tokenizer
tokenizer.save_pretrained(save_path_folder)

# ------------------------------------

"""
