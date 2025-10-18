from passlib.context import CryptContext
from typing import Tuple, Optional

# Context for password hashing
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def is_password_strong(password: str) -> Tuple[bool, Optional[str]]:
    if len(password) < 8:
        return False, "La password deve essere di almeno 8 caratteri"
    if not any(c.isupper() for c in password):
        return False, "La password deve contenere almeno una lettera maiuscola"
    if not any(c.islower() for c in password):
        return False, "La password deve contenere almeno una lettera minuscola"
    if not any(c.isdigit() for c in password):
        return False, "La password deve contenere almeno un numero"
    return True, None
