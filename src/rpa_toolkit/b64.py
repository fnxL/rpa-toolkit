import base64


def is_base64(data: str, encoding: str = "utf-8") -> bool:
    """Check if the provided string is valid base64."""
    if not isinstance(data, str):
        return False

    # Length of base64 string should be a multiple of 4
    if len(data) % 4 != 0:
        return False

    # Attempt to decode the string
    try:
        decoded = base64.b64decode(data, validate=True)
        # Re-encode the decoded data and compare with the original data
        return base64.b64encode(decoded).decode(encoding) == data
    except (base64.binascii.Error, ValueError):
        return False
