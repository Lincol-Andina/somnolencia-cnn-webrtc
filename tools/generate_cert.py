"""generate_cert.py
=============================================================================
Genera un certificado HTTPS autofirmado local para permitir el acceso a la
cámara del celular (getUserMedia exige un contexto seguro).

Uso:
    python tools/generate_cert.py
    python tools/generate_cert.py --force

Los archivos se guardan en ``certs/cert.pem`` y ``certs/key.pem`` (ignorados
por git). Incluye SAN para localhost, 127.0.0.1, el nombre del equipo y la IP
LAN detectada, de modo que el celular pueda abrir https://<IP_LAN>:8501.
=============================================================================
"""

from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import socket
import sys
from pathlib import Path

try:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
except Exception as exc:  # pragma: no cover
    print(f"ERROR: falta la dependencia 'cryptography': {exc}")
    print("Instala con: pip install 'cryptography>=44.0.0'")
    sys.exit(1)


ROOT = Path(__file__).resolve().parent.parent
CERT_DIR = ROOT / "certs"
CERT_PATH = CERT_DIR / "cert.pem"
KEY_PATH = CERT_DIR / "key.pem"

sys.path.insert(0, str(ROOT))
from webrtc_live import get_lan_ips  # noqa: E402


def build_certificate(lan_ips):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "EC"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "GuardDrive AI"),
            x509.NameAttribute(NameOID.COMMON_NAME, "GuardDrive AI Local"),
        ]
    )

    san_entries = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
    ]

    hostname = socket.gethostname()
    if hostname:
        san_entries.append(x509.DNSName(hostname))

    for lan_ip in lan_ips:
        try:
            entry = x509.IPAddress(ipaddress.ip_address(lan_ip))
        except ValueError:
            continue
        if entry not in san_entries:
            san_entries.append(entry)

    now = dt.datetime.now(dt.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )
    return key, cert


def main() -> None:
    parser = argparse.ArgumentParser(description="Certificado HTTPS autofirmado para GuardDrive AI")
    parser.add_argument("--force", action="store_true", help="Sobrescribir certificados existentes")
    parser.add_argument("--out-dir", type=str, default=str(CERT_DIR), help="Directorio de salida")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cert_path = out_dir / "cert.pem"
    key_path = out_dir / "key.pem"

    if (cert_path.exists() or key_path.exists()) and not args.force:
        print(f"Ya existen certificados en {out_dir}.")
        print("Usa --force para regenerarlos.")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        lan_ips = get_lan_ips()
        key, cert = build_certificate(lan_ips)

        key_path.write_bytes(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        print(f"[OK] Certificado: {cert_path}")
        print(f"[OK] Clave privada: {key_path}")

    lan_ips = get_lan_ips()
    primary_ip = lan_ips[0] if lan_ips else "127.0.0.1"
    print("")
    print("Abre en el celular (misma WiFi):")
    for ip in lan_ips or [primary_ip]:
        print(f"  https://{ip}:8501")
    print("")
    print("Lanza la app con:")
    print(
        "  streamlit run app.py --server.address 0.0.0.0 --server.port 8501 "
        "--server.sslCertFile certs/cert.pem --server.sslKeyFile certs/key.pem"
    )
    print("")
    print("Si Windows bloquea el puerto, ejecuta PowerShell como administrador:")
    print(
        '  New-NetFirewallRule -DisplayName "Streamlit DDD 8501" '
        "-Direction Inbound -Protocol TCP -LocalPort 8501 -Action Allow"
    )


if __name__ == "__main__":
    main()
