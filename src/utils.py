import os


def check_and_create_path(path: str):
    """
    Проверяет существование пути и создает его, если он не существует.

    :param path: Путь, который нужно проверить и создать
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print(f"Путь {path} был успешно создан.")
        except Exception as e:
            print(f"Не удалось создать путь {path}: {e}")


