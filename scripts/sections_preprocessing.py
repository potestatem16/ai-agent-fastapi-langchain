import re
import json

def split_markdown_by_headings(markdown_text, include_overview=False):
    """
    Splits the markdown document into sections using headings starting with '#'.

    Parameters:
      - markdown_text: A string containing the markdown content.
      - include_overview (bool): If True, the text before the first heading is stored 
        with the key "overview". Defaults to False.

    For each section, the first line (heading) is extracted and used as the key
    (converted to lowercase with spaces replaced by underscores) for the corresponding
    content, which includes both the heading and the body of the section.
    """
    # Split the document using headings (multiline mode)
    sections = re.split(r'(?m)^#\s+', markdown_text)
    # Remove empty sections and strip whitespace
    sections = [s.strip() for s in sections if s.strip()]
    
    result = {}
    
    # Optionally include an overview section (text before the first heading)
    if include_overview and sections:
        result["overview"] = sections[0]
        sections_to_process = sections[1:]
    else:
        sections_to_process = sections

    # Process each section
    for sec in sections_to_process:
        lines = sec.splitlines()
        if not lines:
            continue
        # The first line is treated as the section heading
        heading = lines[0].strip()
        content = "\n".join(lines[1:]).strip()
        # Combine heading with content
        full_content = heading + "\n" + content if content else heading

        # Create a key from the heading: lowercase and underscores instead of spaces
        key = heading.lower().replace(" ", "_")
        result[key] = full_content

    return result

if __name__ == '__main__':
    # Change this path to the location of your markdown text file (TravelPlanner explanation)
    file_path = r"data\TravelPlanner-info.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # Split the document; set include_overview to True if you want to capture text before the first heading
    sections_dict = split_markdown_by_headings(markdown_text, include_overview=False)

    # Save the sections as a JSON file
    with open(r".\data\TravelPlanner_sections.json", "w", encoding="utf-8") as f:
        json.dump(sections_dict, f, ensure_ascii=False, indent=2)
