from transformers import AutoTokenizer, pipeline,  AutoModelForTokenClassification
from seqeval.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
from seqeval.scheme import IOB2
from dataset import *
device = "cuda" if torch.cuda.is_available() else "cpu"

id2label = {0: 'O',1: 'I-ORG', 2: 'I-PER', 3: 'I-LOC', 4: 'B-ORG', 5: 'B-PER', 6: 'B-LOC'}
label2id = {v:i for i,v in id2label.items()}

def validate(model, data_loader, evaluate=False,lang="en"):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids, atn_mask, labels = batch
            outputs = model(input_ids=input_ids, attention_mask=atn_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            prediction = outputs.logits.argmax(dim=2).cpu().numpy()
            
            cropped_predictions = [pred[1:len(mask) - 1 - mask.flip(dims=[0]).argmax()] for pred, mask in zip(prediction, atn_mask)]
            cropped_true_labels = [true[1:len(mask) - 1 - mask.flip(dims=[0]).argmax()] for true, mask in zip(labels.cpu().numpy(), atn_mask)]

            predictions.extend(cropped_predictions)
            true_labels.extend(cropped_true_labels)

    avg_loss = total_loss / len(data_loader)

    # Convert predictions and true labels to tag sequences
    id2label = {v: k for k, v in label2id.items()}
    pred_tags = [[id2label[tag_id] for tag_id in seq] for seq in predictions]
    true_tags = [[id2label[tag_id] if tag_id in id2label else '-100' for tag_id in seq] for seq in true_labels]
    f1_score_ = f1_score(true_tags, pred_tags)
    if evaluate:
        print(f"LANG [{lang}]: F1-score (Token-level) : {f1_score_:.3f}")
        print(f'Test Loss: {avg_loss} \n Test Evaluation (strict IOB2):')
        print(classification_report(true_tags, pred_tags, zero_division=0, mode='strict', scheme=IOB2, digits=3))
    return avg_loss, true_tags, pred_tags, f1_score_


def plot_loss(train_losses, val_losses, f1_scores, save_path):
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o', label='Validation Loss')
    plt.plot(range(1, len(f1_scores) + 1), f1_scores, marker='.', label='Val. F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.savefig(save_path + 'loss.png')
    plt.show()
    plt.clf()



def train(model, train_loader, val_loader, save_path, epochs=3, lr=5e-5, early_stopping_patience=3, finetune=False, ):
    model.requires_grad_(finetune)
    model.classifier.requires_grad_(True)
    model.train()
    print("Trainable parameters: ", format(sum(p.numel() for p in model.parameters() if p.requires_grad), ","))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    train_losses = []
    val_losses = []
    val_f1s =[]
    
    best_val_loss, best_epoch = float('inf'), 0
    epochs_without_improvement = 0
    best_model_state_dict = None

    for epoch in range(epochs):
        total_train_loss = 0

        # Training
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
        for batch in progress_bar:
            model.train()
            input_ids, atn_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=atn_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            progress_bar.set_postfix({'train_loss': loss.item()})
        train_losses.append(total_train_loss / len(train_loader))

        # Validation
        val_loss, true_tags, pred_tags, f1_score = validate(model, val_loader)
        val_f1s.append(f1_score)
        val_losses.append(val_loss)
        print(f'Epoch {epoch + 1}/{epochs}, train_loss : {loss.item():.4f}, val_loss: {val_loss:.4f}, val_f1: {f1_score:.4f}')
        scheduler.step(val_loss)
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            epochs_without_improvement = 0
            best_model_state_dict = model.state_dict()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch + 1} as validation loss has not decreased for {early_stopping_patience} epochs since epoch-{best_epoch}.')
                break

    # Load the best model state
    model.load_state_dict(best_model_state_dict)

    plot_loss(train_losses, val_losses,val_f1s, save_path)
    return model


def train_test(train_langs, val_langs, model_path, save_path, epochs=20, lr=1e-5, bs=64, finetune=True, model_max_length=256, test_size=0.2):
    # tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=model_max_length)
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=model_max_length,    add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=len(label2id), ignore_mismatched_sizes=True).to(device)

    X_train, X_val, y_train, y_val = combine_and_split_datasets(train_langs, test_size=test_size)
    train_loader = DataLoader(NERDataset(X_train, y_train, tokenizer, label2id), batch_size=bs, shuffle=True, collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id))
    val_loader = DataLoader(NERDataset(X_val, y_val, tokenizer, label2id), batch_size=bs, shuffle=False, collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id))
    model = train(model, train_loader, val_loader, epochs=epochs, lr=lr, finetune=finetune, save_path= save_path)

    for lang, data in getTestDatasets(val_langs=val_langs).items():
        X_test, y_test = data
        test_loader = DataLoader(NERDataset(X_test, y_test, tokenizer, label2id), batch_size=bs, shuffle=False, collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id))
        test_loss, true_tags, pred_tags, _ = validate(model, test_loader, evaluate=True, lang=lang)
    return model


def plot_loss_multi(train_losses, val_losses, f1_scores, name_val_loaders, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Train Loss')
    
    # Plot each validation loss curve
    for i, v_losses in enumerate(val_losses):
        plt.plot(range(1, len(v_losses) + 1), v_losses, marker='o', label=f'{name_val_loaders[i]} Loss')
    
    # Calculate and plot the average validation loss
    # avg_val_losses = [sum(x)/len(x) for x in zip(*val_losses)]
    # plt.plot(range(1, len(avg_val_losses) + 1), avg_val_losses, marker='o', linestyle='--', label='Avg. Validation Loss')

    plt.plot(range(1, len(f1_scores) + 1), f1_scores, marker='.', label='Val. F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/F1 Score')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.savefig(save_path + 'loss.png')
    plt.show()
    plt.clf()

def train_multi(model, train_loader, val_loaders, save_path, name_val_loaders, epochs=3, lr=5e-5, early_stopping_patience=3, finetune=False, ):
    model.requires_grad_(finetune)
    model.classifier.requires_grad_(True)
    model.train()
    print("Trainable parameters: ", format(sum(p.numel() for p in model.parameters() if p.requires_grad), ","))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    train_losses = []
    val_losses = [[] for _ in val_loaders]  # List of lists to track each validation loader
    val_f1s = []  # You might want to adjust this if handling multiple F1 scores
    
    best_val_loss, best_epoch = float('inf'), 0
    epochs_without_improvement = 0
    best_model_state_dict = None

    for epoch in range(epochs):
        total_train_loss = 0
        # Training
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False)
        for batch in progress_bar:
            model.train()
            input_ids, atn_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=atn_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            progress_bar.set_postfix({'train_loss': loss.item()})
        train_losses.append(total_train_loss / len(train_loader))

        # Validation
        for idx, val_loader in enumerate(val_loaders):
            val_loss, true_tags, pred_tags, f1_score = validate(model, val_loader)
            val_losses[idx].append(val_loss)
            # Handling of F1 scores would go here

        # Using the first one since it's the actual validation and not test set
        avg_val_loss = val_losses[0][-1]
        print(val_losses)
        print(avg_val_loss)
        print(f'Epoch {epoch + 1}/{epochs}, train_loss: {loss.item():.4f}, avg_val_loss: {avg_val_loss:.4f}')

        scheduler.step(avg_val_loss)
        
        # Early stopping check based on average validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss, best_epoch = avg_val_loss, epoch
            epochs_without_improvement = 0
            best_model_state_dict = model.state_dict()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f'Early stopping at epoch {epoch + 1} as avg validation loss has not decreased for {early_stopping_patience} epochs since epoch-{best_epoch}.')
                break

    # Load the best model state
    model.load_state_dict(best_model_state_dict)

    plot_loss_multi(train_losses, val_losses,val_f1s, name_val_loaders, save_path)
    return model

def train_test_sep_val(train_langs, val_langs, model_path, save_path, epochs=20, lr=1e-5, bs=64, finetune=True, model_max_length=256, test_size=0.2):
    # tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=model_max_length)
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=model_max_length,    add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=len(label2id), ignore_mismatched_sizes=True).to(device)

    X_train, X_val, y_train, y_val = combine_and_split_datasets(train_langs, test_size=test_size)
    train_loader = DataLoader(NERDataset(X_train, y_train, tokenizer, label2id), batch_size=bs, shuffle=True, collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id))
    val_loader = DataLoader(NERDataset(X_val, y_val, tokenizer, label2id), batch_size=bs, shuffle=False, collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id))
    
    test_loaders = [val_loader]
    name_loaders = ["Real Validation"]
    for lang, data in getTestDatasets(val_langs=val_langs).items():
        X_test, y_test = data
        test_loader = DataLoader(NERDataset(X_test, y_test, tokenizer, label2id), batch_size=bs, shuffle=False, collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id))
        test_loaders.append(test_loader)
        name_loaders.append(lang)
    
    model = train_multi(model, train_loader, test_loaders, epochs=epochs, lr=lr, finetune=finetune, name_val_loaders = name_loaders, save_path= save_path)

    
    for lang, data in getTestDatasets(val_langs=val_langs).items():
        X_test, y_test = data
        test_loader = DataLoader(NERDataset(X_test, y_test, tokenizer, label2id), batch_size=bs, shuffle=False, collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id))
        test_loss, true_tags, pred_tags, _ = validate(model, test_loader, evaluate=True, lang=lang)
    return model