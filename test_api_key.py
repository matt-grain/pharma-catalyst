"""Quick test for Google API key."""

import os
import sys

def test_google_api():
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set")
        print("Run: export GOOGLE_API_KEY='your-key-here'")
        sys.exit(1)

    print(f"API key found: {api_key[:10]}...{api_key[-4:]}")

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        # List available models
        print("\n" + "=" * 50)
        print("AVAILABLE MODELS")
        print("=" * 50)

        models = list(client.models.list())
        gemini_models = [m for m in models if "gemini" in m.name.lower()]

        for model in sorted(gemini_models, key=lambda x: x.name):
            print(f"  - {model.name}")

        print(f"\nTotal: {len(gemini_models)} Gemini models available")

        # Test with gemini-1.5-flash
        print("\n" + "=" * 50)
        print("TESTING API")
        print("=" * 50)

        test_model = "gemini-3-flash-preview"
        response = client.models.generate_content(
            model=test_model,
            contents="Say 'API key works!' and nothing else."
        )

        print(f"Model: {test_model}")
        print(f"Response: {response.text}")
        print("\nGoogle API key is valid!")

    except ImportError:
        print("google-genai not installed. Run: uv add google-genai")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_google_api()
