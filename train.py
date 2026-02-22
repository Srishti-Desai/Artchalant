if _name_ == "_main_":
    X, y = load_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = build_model()

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=4
    )

    model.save("model.h5")
    print("Model training complete and saved as model.h5")
