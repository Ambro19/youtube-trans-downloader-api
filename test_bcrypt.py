from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Test hash and verify
password = "test123"
hash = pwd_context.hash(password)

print("Generated Hash:", hash)
print("Verify correct password:", pwd_context.verify("test123", hash))  # ✅ Should print True
print("Verify wrong password:", pwd_context.verify("wrongpassword", hash))  # ❌ Should print False
