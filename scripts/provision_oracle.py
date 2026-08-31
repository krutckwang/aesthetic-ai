"""
Oracle Cloud Always Free ARM Instance Auto-Provisioner
------------------------------------------------------
Retries all availability domains every 60 seconds until
capacity is found and the instance is created.

Prerequisites:
  1. pip install oci
  2. ~/.oci/config created with your API key (see setup guide)

Run:
  python scripts/provision_oracle.py
"""

import sys
import time
from pathlib import Path

import oci
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

# ── Instance settings (Always Free maximums) ─────────────────────────────────
SHAPE = "VM.Standard.A1.Flex"
OCPUS = 4
MEMORY_GB = 24
DISPLAY_NAME = "aesthetic-ai-server"
RETRY_SECONDS = 60

# ── SSH key paths (auto-generated if missing) ─────────────────────────────────
KEY_DIR = Path.home() / ".oci"
PRIVATE_KEY_FILE = KEY_DIR / "aesthetic_ai_instance.pem"
PUBLIC_KEY_FILE = KEY_DIR / "aesthetic_ai_instance.pub"


# ─────────────────────────────────────────────────────────────────────────────

def get_or_create_ssh_keypair() -> str:
    """Return the SSH public key string, generating a new pair if needed."""
    if PRIVATE_KEY_FILE.exists() and PUBLIC_KEY_FILE.exists():
        print(f"Using existing SSH key: {PRIVATE_KEY_FILE}")
        return PUBLIC_KEY_FILE.read_text().strip()

    print("Generating new SSH key pair...")
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_KEY_FILE.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.OpenSSH,
            serialization.NoEncryption(),
        )
    )
    pub_text = private_key.public_key().public_bytes(
        serialization.Encoding.OpenSSH,
        serialization.PublicFormat.OpenSSH,
    ).decode()
    PUBLIC_KEY_FILE.write_text(pub_text)
    print(f"  Private key: {PRIVATE_KEY_FILE}")
    print(f"  Public key:  {PUBLIC_KEY_FILE}")
    return pub_text


def find_ubuntu_arm_image(compute_client, compartment_id: str) -> str:
    """Return the OCID of the latest Ubuntu 22.04 ARM image."""
    for version in ("22.04", "20.04"):
        images = compute_client.list_images(
            compartment_id,
            operating_system="Canonical Ubuntu",
            operating_system_version=version,
            shape=SHAPE,
            sort_by="TIMECREATED",
            sort_order="DESC",
        ).data
        if images:
            print(f"Using image: {images[0].display_name}")
            return images[0].id
    sys.exit("ERROR: No Ubuntu ARM image found in this region.")


def find_public_subnet(network_client, compartment_id: str) -> str:
    """Return the OCID of the first public subnet found."""
    vcns = network_client.list_vcns(compartment_id).data
    if not vcns:
        sys.exit(
            "ERROR: No VCN found.\n"
            "Go to OCI Console → Networking → Virtual Cloud Networks and create one first."
        )
    for vcn in vcns:
        subnets = network_client.list_subnets(compartment_id, vcn_id=vcn.id).data
        for subnet in subnets:
            if not subnet.prohibit_internet_ingress:
                print(f"Using subnet: {subnet.display_name}")
                return subnet.id
    sys.exit(
        "ERROR: No public subnet found.\n"
        "All subnets in your VCN are private. Create a public subnet first."
    )


def try_all_ads(
    compute_client,
    identity_client,
    compartment_id: str,
    ssh_pub_key: str,
    image_id: str,
    subnet_id: str,
) -> bool:
    """Try every availability domain. Returns True if an instance was created."""
    ads = identity_client.list_availability_domains(compartment_id).data
    for ad in ads:
        print(f"  Trying {ad.name} ...", end=" ", flush=True)
        try:
            response = compute_client.launch_instance(
                oci.core.models.LaunchInstanceDetails(
                    compartment_id=compartment_id,
                    availability_domain=ad.name,
                    display_name=DISPLAY_NAME,
                    shape=SHAPE,
                    shape_config=oci.core.models.LaunchInstanceShapeConfigDetails(
                        ocpus=OCPUS,
                        memory_in_gbs=MEMORY_GB,
                    ),
                    create_vnic_details=oci.core.models.CreateVnicDetails(
                        subnet_id=subnet_id,
                        assign_public_ip=True,
                    ),
                    source_details=oci.core.models.InstanceSourceViaImageDetails(
                        source_type="image",
                        image_id=image_id,
                    ),
                    metadata={"ssh_authorized_keys": ssh_pub_key},
                )
            )
            print("SUCCESS!")
            print()
            print("=" * 60)
            print("Instance created!")
            print(f"  OCID:  {response.data.id}")
            print(f"  State: {response.data.lifecycle_state}")
            print()
            print("Wait 2-3 minutes, then find your Public IP at:")
            print("  OCI Console → Compute → Instances → aesthetic-ai-server")
            print()
            print(f"SSH private key: {PRIVATE_KEY_FILE}")
            print("Connect with:")
            print(f"  ssh -i {PRIVATE_KEY_FILE} ubuntu@<YOUR_PUBLIC_IP>")
            print("=" * 60)
            return True
        except oci.exceptions.ServiceError as e:
            if "Out of capacity" in str(e) or "InternalError" in str(e):
                print("out of capacity")
            elif "LimitExceeded" in str(e):
                print("LIMIT EXCEEDED — you may already have an instance running")
                sys.exit("Check OCI Console → Compute → Instances")
            else:
                print(f"error: {e.message}")
    return False


def main() -> None:
    print("Oracle Cloud ARM Instance Auto-Provisioner")
    print("=" * 60)

    # Load and validate OCI config
    try:
        config = oci.config.from_file()
        oci.config.validate_config(config)
    except Exception as exc:
        sys.exit(
            f"OCI config error: {exc}\n"
            "Complete Step 2 first: create ~/.oci/config with your API key."
        )

    compartment_id = config["tenancy"]
    print(f"Tenancy: {compartment_id}")
    print(f"Region:  {config['region']}")
    print()

    compute_client = oci.core.ComputeClient(config)
    identity_client = oci.identity.IdentityClient(config)
    network_client = oci.core.VirtualNetworkClient(config)

    # Corporate SSL inspection workaround — disable certificate verification
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    for client in (compute_client, identity_client, network_client):
        client.base_client.session.verify = False

    ssh_pub_key = get_or_create_ssh_keypair()

    print("\nLooking up Ubuntu 22.04 ARM image...")
    image_id = find_ubuntu_arm_image(compute_client, compartment_id)

    print("Looking up public subnet...")
    subnet_id = find_public_subnet(network_client, compartment_id)

    print(f"\nRetrying every {RETRY_SECONDS}s across all availability domains.")
    print("Leave this window open. Press Ctrl+C to stop.\n")

    attempt = 0
    while True:
        attempt += 1
        print(f"[Attempt {attempt}]  {time.strftime('%H:%M:%S')}")
        if try_all_ads(
            compute_client, identity_client,
            compartment_id, ssh_pub_key, image_id, subnet_id,
        ):
            break
        print(f"  All ADs full. Waiting {RETRY_SECONDS}s...\n")
        time.sleep(RETRY_SECONDS)


if __name__ == "__main__":
    main()
