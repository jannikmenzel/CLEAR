'''loading packages'''
import  argparse

'''Defining parameters about our method'''
def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal electronic healthcare records anaysis")
    
    parser.add_argument("--data_path", type=str, default="path to /CLEAR/EHRs/data/dataset/", help="A path to dataset folder")
    parser.add_argument("--output_path", type=str, default="./weights/CLEAR", help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    
    parser.add_argument("--task", type=str, default="critical_outcome",
                        choices=["hospitalization", "inhospital_mortality", "icu_transfer_12h", "critical_outcome", "odir_paper"])
    parser.add_argument("--modalities", nargs='+', default=['category_attributes', 'numerical_sequence', 'category_sequence1', 'category_sequence2', 'notes'])
    parser.add_argument("--num_modalities", type=int, default=5, help="The number of modalities.")
    parser.add_argument("--dim_triage", type=int, default=65, help="The dimension of triage variables.")
    parser.add_argument("--dim_labtest", type=int, default=7, help="The dimension of labtest.")
    parser.add_argument("--dim_medications", type=int, default=704, help="The dimension of medications.")
    parser.add_argument("--dim_diagnoses", type=int, default=10321, help="The dimension of diagnoses.")
    parser.add_argument("--dim_notes", type=int, default=768, help="The dimension of medical notes.")
    parser.add_argument("--num_train_epochs", type=int, default=50, help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size  for the training dataloader.")
    parser.add_argument("--eval_batch_size", type=int, default=1024, help="Batch size for the evaluation dataloader.")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Initial learning rate to use.")
    # parser.add_argument("--learning_rate", type=float, default=2e-4, help="Initial learning rate to use.")
    parser.add_argument("--txt_learning_rate", type=float, default=2e-5, help="Initial learning rate to use.")
    # parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay to use.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay to use.")
    parser.add_argument("--tao", type=float, default=0.3, help="Weight decay to use.")
    parser.add_argument("--alpha", type=float, default=0.4, help="Weight decay to use.")
    
    parser.add_argument("--language_model", default='ClinicalBERT', type=str, help="model for text")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads.")
    parser.add_argument("--layers", type=int, default=2, help="Number of transformer encoder layer.")
    parser.add_argument("--cross_layers", type=int, default=2, help="Number of transformer cross encoder layer.")
    parser.add_argument("--embed_dim", default=128, type=int, help="attention embedding dim.")
    parser.add_argument("--dropout", default=0.10, type=float, help="dropout.")
    parser.add_argument('--num_of_notes', help='Number of notes to include for a patient input 0 for all the notes', type=int, default=5)
    parser.add_argument('--num_of_labtests', help='Number of labtest to include for a patient input 0 for all the notes', type=int, default=12)
    parser.add_argument("--max_length", type=int, default=128, 
                        help=("The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"\
                              " sequences shorter will be padded if `--pad_to_max_lengh` is passed."))
    parser.add_argument('--num_labels', type=int, default=2)    

    parser.add_argument( "--pad_to_max_length", action="store_true", help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.")
    parser.add_argument('--notes_order', 
                        help='Should we get notes from beginning of the admission time or from end of it, outputions are: \
                        1. First: pick first notes 2. Last: pick last notes', default=None)
    
    parser.add_argument("--mode", type=str, default="train", help="train, val or test")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--warmup_proportion", default=0.10, type=float, help="proportion for the warmup in the lr scheduler.")
    parser.add_argument("--pretrained_model_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument('--fine_tune', type=str, default="False")
    
    args = parser.parse_args()
    return args