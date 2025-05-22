from data import MyDataModule  # исправь импорт под свой проект

if __name__ == "__main__":
    data_dir = "C:/Users/User/Desktop/MLOps/Data/1" 

    dm = MyDataModule(data_dir=data_dir, batch_size=4)
    dm.setup(stage="fit")

    train_loader = dm.train_dataloader()

    batch = next(iter(train_loader))

    features = batch['features']
    labels = batch['label']
    print('new')
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)

    print("Пример features (первые 2 строки):")
    print(features[:2])

    print("Пример labels:")
    print(labels)
