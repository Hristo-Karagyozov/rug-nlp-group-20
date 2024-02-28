import torch
import transformers


class EmoClassifier(torch.nn.Module):
    def __init__(self, model, output_dim):
        super(EmoClassifier, self).__init__()
        self.pretrained_model = transformers.RobertaModel.from_pretrained(model)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, output_dim),
            torch.nn.Softmax()
        )

    def forward(self, ids, attention_mask, token_type_ids):
        pretrained_output = self.pretrained_model(ids, attention_mask, token_type_ids)
        return self.classifier_head(pretrained_output)
