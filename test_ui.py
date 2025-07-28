import subprocess

def run_streamlit():
    print("Lancement de l'interface Streamlit...")
    result = subprocess.run(["streamlit", "run", "src/streamlit_app.py"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Streamlit UI launched successfully.")
    else:
        print("❌ Streamlit UI failed to launch.")
        print("---- STDERR ----")
        print(result.stderr)
        print("---- STDOUT ----")
        print(result.stdout)

if __name__ == "__main__":
    run_streamlit()