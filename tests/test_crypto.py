from encryption.crypto import (
    CryptoIntegrityError,
    CryptoManager,
)


def main():

    # ---------------------------------------------------------
    # Generate key
    # ---------------------------------------------------------

    key = CryptoManager.generate_key()

    print(
        "Key size:",
        len(key),
    )

    # ---------------------------------------------------------
    # Encode key
    # ---------------------------------------------------------

    encoded = CryptoManager.encode_key(
        key
    )

    print(
        "Base64 key:",
        encoded,
    )

    restored_key = CryptoManager.decode_key(
        encoded
    )

    assert restored_key == key

    # ---------------------------------------------------------
    # Encrypt
    # ---------------------------------------------------------

    plaintext = (
        "Secure RAG confidential document."
    )

    payload = CryptoManager.encrypt_text(
        plaintext,
        key,
        associated_data=b"secure-rag",
    )

    print(
        "Ciphertext size:",
        len(payload.ciphertext),
    )

    # ---------------------------------------------------------
    # Decrypt
    # ---------------------------------------------------------

    result = CryptoManager.decrypt_text(
        payload,
        key,
        associated_data=b"secure-rag",
    )

    assert result == plaintext

    print(
        "Encryption/decryption: OK"
    )

    # ---------------------------------------------------------
    # Tamper detection
    # ---------------------------------------------------------

    corrupted = bytearray(
        payload.ciphertext
    )

    corrupted[-1] ^= 0x01

    corrupted_payload = type(
        payload
    )(
        nonce=payload.nonce,
        ciphertext=bytes(
            corrupted
        ),
    )

    try:

        CryptoManager.decrypt_text(
            corrupted_payload,
            key,
            associated_data=b"secure-rag",
        )

    except CryptoIntegrityError:

        print(
            "Tamper detection: OK"
        )


if __name__ == "__main__":
    main()