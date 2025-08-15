import os
import json
import time
import fitz  # PyMuPDF
import streamlit as st
import google.generativeai as genai
from collections import defaultdict

# --- Gemini API Setup ---
def get_api_key():
    """Get API key from various sources with proper error handling."""
    # Try environment variable first
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return api_key
    
    # Try Streamlit secrets
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        return api_key
    except (KeyError, FileNotFoundError):
        pass
    
    return None

api_key = get_api_key()
if not api_key:
    st.error("🔑 **GEMINI_API_KEY not found!**")
    st.markdown("""
    Please set up your Gemini API key using one of these methods:
    
    **Option 1: Environment Variable (Recommended for local development)**
    ```bash
    set GEMINI_API_KEY=your_api_key_here
    ```
    
    **Option 2: Streamlit Secrets**
    Create a file `.streamlit/secrets.toml` in your project directory:
    ```toml
    GEMINI_API_KEY = "your_api_key_here"
    ```
    
    **Get your API key from:** https://makersuite.google.com/app/apikey
    """)
    st.stop()

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}")
    st.stop()

# --- Chunk Extraction (Improved Version) ---
def group_blocks_into_paragraphs(blocks, min_words_per_para=15):
    """
    Groups raw text blocks from PyMuPDF into semantically meaningful paragraphs.
    A new paragraph is started if a block ends with a sentence-terminating punctuation
    or if it looks like a list item that should be grouped with the previous text.
    """
    paragraphs = []
    current_para = ""

    for b in blocks:
        # Clean the text block
        text = b[4].replace('\n', ' ').strip()
        if not text:
            continue

        # Check if the block looks like a list item (bullet, number, etc.)
        is_list_item = text.startswith(('•', '*', '-', '–')) or (len(text) > 2 and text[0].isdigit() and text[1] in '.)')

        # If the current paragraph is not empty and the new block starts
        # with a lowercase letter and is not a list item, it's likely a continuation.
        if current_para and text and text[0].islower() and not is_list_item:
            current_para += " " + text
        # If the previous paragraph ended without punctuation, and this new block is a list item
        # or seems related, group them together.
        elif current_para and not current_para.endswith(('.', '?', '!')) :
             current_para += " " + text
        # Otherwise, start a new paragraph
        else:
            if current_para and len(current_para.split()) >= min_words_per_para:
                paragraphs.append(current_para)
            current_para = text
    
    # Add the last processed paragraph if it's substantial
    if current_para and len(current_para.split()) >= min_words_per_para:
        paragraphs.append(current_para)
        
    return paragraphs

def extract_semantic_chunks(pdf_file, min_words=25, max_words=700):
    """
    Extracts text from a PDF and intelligently combines it into semantically meaningful chunks.
    This version is designed to group related lines and list items together.
    """
    all_paragraphs = []
    try:
        pdf_file.seek(0)
        pdf_bytes = pdf_file.read()
        
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_num, page in enumerate(doc):
                try:
                    # Get blocks, filtering out non-text (e.g., images)
                    blocks = [b for b in page.get_text("blocks") if len(b) > 6 and b[6] == 0]
                    
                    if not blocks:
                        continue

                    page_paras = group_blocks_into_paragraphs(blocks, min_words)
                    
                    # Smartly merge paragraphs across page breaks
                    if all_paragraphs and page_paras:
                        last_para = all_paragraphs[-1]
                        first_para = page_paras[0]
                        # If last paragraph on previous page seems incomplete...
                        if not last_para.endswith(('.', '!', '?')):
                           # ...and first paragraph on new page continues it.
                           all_paragraphs[-1] += " " + first_para
                           all_paragraphs.extend(page_paras[1:])
                        else:
                            all_paragraphs.extend(page_paras)
                    else:
                        all_paragraphs.extend(page_paras)
                except Exception as e:
                    st.warning(f"⚠️ Error processing page {page_num + 1}: {e}")
                    continue
                    
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return []

    # Final filtering and splitting based on max_words
    final_chunks = []
    for para in all_paragraphs:
        if len(para.split()) > max_words:
            # If a paragraph is too long, split it. This is a simple fallback.
            # A more advanced method could use sentence tokenizers.
            parts = [para[i:i+max_words] for i in range(0, len(para), max_words)]
            final_chunks.extend(parts)
        else:
            final_chunks.append(para)

    return [chunk for chunk in final_chunks if len(chunk.split()) >= min_words]


# --- Tag Extraction ---
def extract_tags_with_gemini(chunk):
    """Calls Gemini API to get primary and secondary tags for a text chunk."""
    prompt = f"""
You are a semantic annotation assistant specializing in document analysis.
Given the following chunk, identify:
1. One primary tag that captures the MAIN FUNCTIONAL CONCEPT (use broader, more general categories)
2. Three to five secondary tags (supporting ideas, specific actions, or sub-themes)

TAGGING PRINCIPLES:
- For recommendations/advice/guidance: Focus on the POLICY DOMAIN or FUNCTIONAL AREA
- For procedures/processes: Use the PROCESS TYPE or OPERATIONAL CATEGORY  
- For discussions/analysis: Use the SUBJECT MATTER or DOMAIN AREA
- Prioritize FUNCTIONAL PURPOSE over descriptive details
- Use consistent terminology across similar concepts

Good primary tag patterns:
- "[Domain] Policy" for any recommendations or guidance
- "[Process] Management" for operational procedures
- "[Subject] Analysis" for discussions or evaluations
- "[Area] Framework" for structural or systematic content

Respond ONLY with a JSON object like:
{{
  "primary_tag": "Functional Category",
  "secondary_tags": ["Sub1", "Sub2", "Sub3"]
}}

chunk:
\"\"\"{chunk}\"\"\"
"""
    try:
        response = model.generate_content(prompt)
        content = response.text.strip()
        # Clean up potential markdown formatting from the response
        json_string = content.replace("```json", "").replace("```", "").strip()
        parsed_response = json.loads(json_string)
        
        # Validate the response structure
        if not isinstance(parsed_response, dict):
            raise ValueError("Response is not a dictionary")
        if "primary_tag" not in parsed_response:
            raise ValueError("Missing primary_tag in response")
        if "secondary_tags" not in parsed_response:
            parsed_response["secondary_tags"] = []
        
        return parsed_response
    except json.JSONDecodeError as e:
        st.warning(f"⚠️ JSON parsing error for chunk: {e}")
        return None
    except Exception as e:
        st.warning(f"⚠️ Could not extract tags for a chunk: {e}")
        return None


# --- Tag Normalization (with Batching) ---
def normalize_tags(tag_list, batch_size=75):
    """Normalizes a list of tags in batches to avoid API token limits."""
    if not tag_list:
        return {}
        
    prompt_template = """
You are an expert at clustering semantically similar concepts for document analysis.
Group the following tags based on their FUNCTIONAL MEANING and CONCEPTUAL SIMILARITY.

NORMALIZATION PRINCIPLES:
- Group tags that refer to the same functional domain or policy area
- Combine variations of the same concept (e.g., "Risk Management", "Risk Control", "Risk Assessment" → "Risk Management")
- Merge similar procedural or operational concepts
- Use the most comprehensive and clear term as the canonical tag
- Focus on FUNCTIONAL EQUIVALENCE rather than exact word matching

Your goal: Ensure that content discussing the same functional area gets the same canonical tag, regardless of specific wording variations.

Respond ONLY with a JSON dictionary mapping each original tag to its canonical tag:
{{ "Risk Assessment": "Risk Management", "Quality Control": "Quality Management", "Safety Protocol": "Safety Management" }}

Tags to normalize:
{tags_json}
"""
    final_mapping = {}
    st.write(f"Normalizing {len(tag_list)} unique tags in batches...")

    for i in range(0, len(tag_list), batch_size):
        batch = tag_list[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(tag_list) + batch_size - 1) // batch_size
        st.write(f"  - Processing batch {batch_num}/{total_batches}...")
        prompt = prompt_template.format(tags_json=json.dumps(batch))

        try:
            response = model.generate_content(prompt)
            content = response.text.strip()
            json_string = content.replace("```json", "").replace("```", "").strip()
            if json_string:
                batch_mapping = json.loads(json_string)
                if isinstance(batch_mapping, dict):
                    final_mapping.update(batch_mapping)
                else:
                    st.warning(f"⚠️ Batch {batch_num} returned non-dictionary response")
        except json.JSONDecodeError as e:
            st.warning(f"⚠️ JSON parsing error in batch {batch_num}: {e}")
        except Exception as e:
            st.warning(f"⚠️ Could not process batch {batch_num}: {e}")

        time.sleep(1)  # Be kind to the API

    return final_mapping


# --- Tag Chunks ---
def tag_chunks(chunks, label=""):
    """Iterates through chunks and gets tags for each one."""
    if not chunks:
        return []
        
    tagged = []
    progress_text = f"Tagging {label}... (This may take several minutes for large documents)"
    progress = st.progress(0, text=progress_text)
    
    for i, chunk in enumerate(chunks):
        tags = extract_tags_with_gemini(chunk)
        if tags:
            tagged.append({
                "chunk_number": i + 1,
                "original_primary_tag": tags.get("primary_tag"),
                "secondary_tags": tags.get("secondary_tags", []),
                "text_snippet": chunk[:200].replace('\n', ' ') + ("..." if len(chunk) > 200 else ""),
                "full_text": chunk,
                "source": label
            })
        time.sleep(1)  # API rate limiting
        progress.progress((i + 1) / len(chunks), text=f"Tagging {label} chunk {i+1}/{len(chunks)}")
    
    progress.empty()
    return tagged


# --- Semantic Comparison ---
def detect_semantic_conflict(chunk_a, chunk_b):
    """Compares two text chunks for meaningful semantic conflicts."""
    prompt = f"""
You are an expert legal and policy analyst comparing two paragraphs. Your task is to identify whether there is a *meaningful semantic conflict* between them.

Detect conflicts including:
- New facts, restrictions, or obligations being introduced
- Claims, rights, or duties being removed or reversed  
- Changes in scope, application, or implication
- Direct contradictions in statements or conclusions
- DIFFERENT POLICY RECOMMENDATIONS or approaches to the same problem
- Conflicting advice or guidance on the same topic

Be especially sensitive to:
- Different recommended actions (e.g., "use public transport" vs "stay home")
- Conflicting strategies or solutions
- Opposite approaches to handling the same situation

Rules for your response:
1. If the paragraphs are about completely different topics, respond with: **UNRELATED**
2. If the paragraphs discuss the same topic but have no meaningful conflict, respond with: **NO_SEMANTIC_CONFLICT**
3. If ANY conflict exists (including different recommendations), summarize the conflict in one precise sentence.

Paragraph A: "{chunk_a[:1000]}{'...' if len(chunk_a) > 1000 else ''}"

Paragraph B: "{chunk_b[:1000]}{'...' if len(chunk_b) > 1000 else ''}"

Response:
"""
    try:
        response = model.generate_content(prompt)
        result = response.text.strip()
        return result if result else "NO_SEMANTIC_CONFLICT"
    except Exception as e:
        return f"ERROR: {e}"


# --- Streamlit UI ---
st.set_page_config("PDF Semantic Cross-Referencer", layout="wide")
st.title("⚔️ Document Concept Finder and Conflict Detector")

col1, col2 = st.columns(2)
with col1:
    uploaded_file1 = st.file_uploader("Upload PDF 1", type="pdf", key="pdf1")
with col2:
    uploaded_file2 = st.file_uploader("Upload PDF 2", type="pdf", key="pdf2")

if uploaded_file1 and uploaded_file2:
    with st.status("Analyzing PDFs...", expanded=True) as status:
        try:
            status.update(label="Step 1/4: Extracting text from documents...")
            # Use the new semantic chunking function
            chunks1 = extract_semantic_chunks(uploaded_file1, min_words=25, max_words=700)
            chunks2 = extract_semantic_chunks(uploaded_file2, min_words=25, max_words=700)

            if not chunks1:
                st.error(f"No valid text chunks found in {uploaded_file1.name}")
                st.stop()
            if not chunks2:
                st.error(f"No valid text chunks found in {uploaded_file2.name}")
                st.stop()

            st.write(f"Extracted {len(chunks1)} chunks from {uploaded_file1.name}")
            st.write(f"Extracted {len(chunks2)} chunks from {uploaded_file2.name}")

            status.update(label="Step 2/4: Tagging content from documents...")
            tagged1 = tag_chunks(chunks1, uploaded_file1.name)
            tagged2 = tag_chunks(chunks2, uploaded_file2.name)
            all_tagged = tagged1 + tagged2

            if not all_tagged:
                st.error("No valid tags were extracted from the PDFs.")
                st.stop()

            status.update(label="Step 3/4: Normalizing primary tags...")
            unique_primary_tags = list(set(
                t["original_primary_tag"] for t in all_tagged 
                if t.get("original_primary_tag")
            ))
            
            if unique_primary_tags:
                primary_tag_mapping = normalize_tags(unique_primary_tags)
                for t in all_tagged:
                    original_tag = t.get("original_primary_tag")
                    t["primary_tag"] = primary_tag_mapping.get(original_tag, original_tag) if original_tag else None
            else:
                st.warning("No primary tags found to normalize")
                for t in all_tagged:
                    t["primary_tag"] = t.get("original_primary_tag")

            # Build search index for PDF 2 using normalized primary tags and original secondary tags
            index2 = defaultdict(list)
            for chunk in tagged2:
                if chunk.get("primary_tag"):
                    index2[chunk["primary_tag"]].append(chunk)
                for sec_tag in chunk.get("secondary_tags", []):
                    if sec_tag:  # Ensure tag is not empty
                        index2[sec_tag].append(chunk)

            # --- Comparison Step with Live Counter ---
            status.update(label="Step 4/4: Comparing for conflicts...")
            comparison_results = []
            compared_pairs = set()
            conflicts_found_counter = 0
            total_chunks = len(tagged1)

            for i, chunk1 in enumerate(tagged1):
                current_status = f"Step 4/4: Comparing chunk {i+1}/{total_chunks} (Conflicts found: {conflicts_found_counter})"
                status.update(label=current_status)

                tags_to_check = set()
                if chunk1.get("primary_tag"):
                    tags_to_check.add(chunk1["primary_tag"])
                tags_to_check.update(chunk1.get("secondary_tags", []))
                
                # Remove empty tags
                tags_to_check = {tag for tag in tags_to_check if tag and tag.strip()}

                for tag in tags_to_check:
                    for chunk2 in index2[tag]:
                        pair_id = tuple(sorted((chunk1["full_text"], chunk2["full_text"])))
                        if pair_id in compared_pairs:
                            continue
                        compared_pairs.add(pair_id)

                        result = detect_semantic_conflict(chunk1["full_text"], chunk2["full_text"])

                        if result != "NO_SEMANTIC_CONFLICT":
                            if result not in ["UNRELATED"] and not result.startswith("ERROR"):
                                conflicts_found_counter += 1

                            comparison_results.append({
                                "match_on_tag": tag,
                                "chunk1_info": {
                                    "number": chunk1["chunk_number"], 
                                    "primary_tag": chunk1["primary_tag"], 
                                    "snippet": chunk1["text_snippet"],
                                    "source": chunk1["source"]
                                },
                                "chunk2_info": {
                                    "number": chunk2["chunk_number"], 
                                    "primary_tag": chunk2["primary_tag"], 
                                    "snippet": chunk2["text_snippet"],
                                    "source": chunk2["source"]
                                },
                                "comparison_result": result,
                                "full_text_1": chunk1["full_text"],
                                "full_text_2": chunk2["full_text"]
                            })
                        time.sleep(1)

            status.update(label="✅ Analysis Complete!", state="complete")

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            st.stop()

    # --- Display Results ---
    st.header("📊 Conflict Detection Results")
    conflicts = [r for r in comparison_results if r["comparison_result"] not in ["NO_SEMANTIC_CONFLICT", "UNRELATED"]]
    unrelated = [r for r in comparison_results if r["comparison_result"] == "UNRELATED"]

    st.subheader(f"🔴 Found {len(conflicts)} Semantic Conflicts")
    if conflicts:
        for r in conflicts:
            with st.container(border=True):
                st.error(f"Conflict on tag: `{r['match_on_tag']}`")
                st.markdown(f"**Summary:** {r['comparison_result']}")
                st.markdown(f"- **{r['chunk1_info']['source']}** (Chunk {r['chunk1_info']['number']}): _{r['chunk1_info']['snippet']}_")
                st.markdown(f"- **{r['chunk2_info']['source']}** (Chunk {r['chunk2_info']['number']}): _{r['chunk2_info']['snippet']}_")
    else:
        st.success("No semantic conflicts were found between the documents.")
    
    if comparison_results:
        st.download_button(
            "⬇️ Download Full JSON Report", 
            json.dumps(comparison_results, indent=2), 
            file_name="conflict_report.json",
            mime="application/json"
        )

    st.subheader(f"⚪️ Found {len(unrelated)} Unrelated Matches on Shared Tags")
    if unrelated:
        with st.expander("Show Unrelated Matches"):
            for r in unrelated:
                st.info(f"Unrelated match on tag `{r['match_on_tag']}`")
                st.markdown(f"- {r['chunk1_info']['source']}: _{r['chunk1_info']['snippet']}_")
                st.markdown(f"- {r['chunk2_info']['source']}: _{r['chunk2_info']['snippet']}_")

    st.header("🔖 All Tagged Chunks")
    with st.expander("Show Tagged Chunks"):
        col1_res, col2_res = st.columns(2)
        with col1_res:
            st.write(f"#### {uploaded_file1.name}")
            for item in tagged1:
                primary_tag = item.get('primary_tag', 'No tag')
                st.markdown(f"**Chunk {item['chunk_number']}** | Tag: `{primary_tag}`")
                st.caption(item["text_snippet"])
        with col2_res:
            st.write(f"#### {uploaded_file2.name}")
            for item in tagged2:
                primary_tag = item.get('primary_tag', 'No tag')
                st.markdown(f"**Chunk {item['chunk_number']}** | Tag: `{primary_tag}`")
                st.caption(item["text_snippet"])