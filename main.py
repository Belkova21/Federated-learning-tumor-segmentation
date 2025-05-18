import torch

from segmentation_Unet import DynamicUNet, BrainTumorClassifier
from dataset import single_data_load
from test import evaluate_model_on_loader


def main():
    # setup
    train_loader, val_loader, test_loader = single_data_load()
    model = DynamicUNet(filters=[32, 64, 128, 256, 512], input_channels=1, output_channels=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = BrainTumorClassifier(model, device)

    #training
    train_loss = trainer.train(
        trainloader=train_loader,
        valloader=val_loader,
        epochs=1,
        mini_batch_log=1,  # napríklad logovať každých 10 batchov
        save_best_path="best_model.pth"  # môžeš uložiť najlepší model
    )

    #evalution
    evaluate_model_on_loader(model,test_loader,device)


if __name__ == "__main__":
    main()


