import bcrypt


def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


MOCK_USERS_DB = {
    "dr_smith": {
        "username": "dr_smith",
        "hashed_password": get_password_hash("doc123"),
        "role": "Doctor"
    },
    "nurse_joy": {
        "username": "nurse_joy",
        "hashed_password": get_password_hash("nurse123"),
        "role": "Nurse"
    }
}
