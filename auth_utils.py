# =================================================================================================
# auth_utils.py - Production-Ready Password Hashing Module
# Fixes bcrypt 72-byte limit issue for production deployment
# =================================================================================================

import hashlib
from passlib.context import CryptContext

# Initialize password context with bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    """
    Hash password with bcrypt, respecting 72-byte limit.
    
    bcrypt has a hard limit of 72 bytes for password length. This function
    handles passwords of any length by pre-hashing long passwords with SHA-256.
    
    Args:
        password (str): Plain text password to hash
        
    Returns:
        str: Bcrypt hashed password
    """
    if not password:
        raise ValueError("Password cannot be empty")
    
    # Convert to bytes for length checking
    password_bytes = password.encode('utf-8')
    
    # If password exceeds bcrypt's 72-byte limit, pre-hash it
    if len(password_bytes) > 72:
        # SHA-256 produces a consistent 64-character hex string
        password_digest = hashlib.sha256(password_bytes).hexdigest()
        return pwd_context.hash(password_digest)
    else:
        # Standard bcrypt for passwords within limit
        return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its bcrypt hash.
    
    Args:
        plain_password (str): Plain text password to verify
        hashed_password (str): Bcrypt hash to verify against
        
    Returns:
        bool: True if password matches hash, False otherwise
    """
    if not plain_password or not hashed_password:
        return False
    
    try:
        password_bytes = plain_password.encode('utf-8')
        
        # Apply same pre-hash logic if password is too long
        if len(password_bytes) > 72:
            password_digest = hashlib.sha256(password_bytes).hexdigest()
            return pwd_context.verify(password_digest, hashed_password)
        else:
            return pwd_context.verify(plain_password, hashed_password)
            
    except Exception as e:
        print(f"Password verification error: {e}")
        return False


def validate_password_strength(password: str) -> tuple:
    """
    Validate password meets security requirements.
    
    Args:
        password (str): Password to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not password:
        return False, "Password cannot be empty"
    
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if len(password) > 128:
        return False, "Password is too long (maximum 128 characters)"
    
    return True, ""


# # =================================================================================================
# # auth_utils.py - Production-Ready Password Hashing Module
# # Fixes bcrypt 72-byte limit issue for production deployment
# # =================================================================================================

# import hashlib
# from passlib.context import CryptContext

# # Initialize password context with bcrypt
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# def get_password_hash(password: str) -> str:
#     """
#     Hash password with bcrypt, respecting 72-byte limit.
    
#     bcrypt has a hard limit of 72 bytes for password length. This function
#     handles passwords of any length by pre-hashing long passwords with SHA-256.
    
#     Security considerations:
#     - Passwords <= 72 bytes: Direct bcrypt hashing (standard)
#     - Passwords > 72 bytes: SHA-256 pre-hash + bcrypt (maintains security)
#     - Pre-hashing doesn't weaken security as SHA-256 output is fixed 64-char hex
    
#     Args:
#         password (str): Plain text password to hash
        
#     Returns:
#         str: Bcrypt hashed password
        
#     Raises:
#         ValueError: If password is empty or None
        
#     Examples:
#         >>> hash1 = get_password_hash("short_password")
#         >>> hash2 = get_password_hash("very_long_password" * 10)
#         >>> # Both work correctly despite different lengths
#     """
#     if not password:
#         raise ValueError("Password cannot be empty")
    
#     # Convert to bytes for length checking
#     password_bytes = password.encode('utf-8')
    
#     # If password exceeds bcrypt's 72-byte limit, pre-hash it
#     if len(password_bytes) > 72:
#         # SHA-256 produces a consistent 64-character hex string (256 bits)
#         # This is always < 72 bytes and maintains entropy
#         password_digest = hashlib.sha256(password_bytes).hexdigest()
#         return pwd_context.hash(password_digest)
#     else:
#         # Standard bcrypt for passwords within limit
#         return pwd_context.hash(password)


# def verify_password(plain_password: str, hashed_password: str) -> bool:
#     """
#     Verify a password against its bcrypt hash.
    
#     Automatically handles both direct-hashed and pre-hashed passwords
#     by applying the same logic used during hashing.
    
#     Args:
#         plain_password (str): Plain text password to verify
#         hashed_password (str): Bcrypt hash to verify against
        
#     Returns:
#         bool: True if password matches hash, False otherwise
        
#     Examples:
#         >>> hashed = get_password_hash("my_password")
#         >>> verify_password("my_password", hashed)
#         True
#         >>> verify_password("wrong_password", hashed)
#         False
#     """
#     if not plain_password or not hashed_password:
#         return False
    
#     try:
#         password_bytes = plain_password.encode('utf-8')
        
#         # Apply same pre-hash logic if password is too long
#         if len(password_bytes) > 72:
#             password_digest = hashlib.sha256(password_bytes).hexdigest()
#             return pwd_context.verify(password_digest, hashed_password)
#         else:
#             return pwd_context.verify(plain_password, hashed_password)
            
#     except Exception as e:
#         # Log but don't expose details
#         print(f"Password verification error: {e}")
#         return False


# def validate_password_strength(password: str) -> tuple[bool, str]:
#     """
#     Validate password meets security requirements.
    
#     Rules:
#     - Minimum 8 characters
#     - Maximum 128 characters (reasonable limit)
#     - At least one letter and one number (optional, can be enforced)
    
#     Args:
#         password (str): Password to validate
        
#     Returns:
#         tuple[bool, str]: (is_valid, error_message)
        
#     Examples:
#         >>> validate_password_strength("short")
#         (False, "Password must be at least 8 characters long")
#         >>> validate_password_strength("ValidPass123")
#         (True, "")
#     """
#     if not password:
#         return False, "Password cannot be empty"
    
#     if len(password) < 8:
#         return False, "Password must be at least 8 characters long"
    
#     if len(password) > 128:
#         return False, "Password is too long (maximum 128 characters)"
    
#     # Optional: Add complexity requirements
#     # has_letter = any(c.isalpha() for c in password)
#     # has_number = any(c.isdigit() for c in password)
#     # if not (has_letter and has_number):
#     #     return False, "Password must contain at least one letter and one number"
    
#     return True, ""


# # =================================================================================================
# # Usage in main.py:
# # 
# # Replace the existing password functions with:
# # 
# # from auth_utils import get_password_hash, verify_password, validate_password_strength
# # 
# # In registration endpoint:
# # is_valid, error_msg = validate_password_strength(user.password)
# # if not is_valid:
# #     raise HTTPException(400, error_msg)
# # hashed_password = get_password_hash(user.password)
# # 
# # In login endpoint:
# # if not verify_password(form_data.password, user.hashed_password):
# #     raise HTTPException(401, "Incorrect username or password")
# # =================================================================================================