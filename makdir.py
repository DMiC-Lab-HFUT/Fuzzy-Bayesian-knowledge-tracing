def mkdir(path):
    import os

    path = path.strip()

    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    if not isExists:

        os.makedirs(path)
        return True
    else:
        return False