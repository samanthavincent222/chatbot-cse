from pypdf import PdfReader

PDF_PATH = "/Users/samanthavincent/Downloads/cse_notes.pdf"

print("Trying to load:", PDF_PATH)

reader = PdfReader(PDF_PATH)

print("✅ PDF opened successfully")
print("Number of pages:", len(reader.pages))

text = reader.pages[0].extract_text()

print("\n--- FIRST 300 CHARACTERS ---\n")
print(text[:300] if text else "❌ No text extracted from first page")
