"""Fix duplicated content in frontend tsx files."""
import os
import re

def fix_duplicated_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check for duplicate 'export default function' pattern
    matches = list(re.finditer(r'^export default function \w+', content, re.MULTILINE))
    if len(matches) >= 2:
        second_match_start = matches[1].start()
        fixed_content = content[:second_match_start].rstrip() + '\n'
        print(f'Fixed: {filepath}')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        return True

    # Check for duplicate 'export default' (const/class) pattern
    matches = list(re.finditer(r'^export default ', content, re.MULTILINE))
    if len(matches) >= 2:
        second_match_start = matches[1].start()
        fixed_content = content[:second_match_start].rstrip() + '\n'
        print(f'Fixed: {filepath}')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        return True

    # Check for duplicate import statements at unusual positions (after code)
    # Find if there's an import statement after an export default
    export_match = re.search(r'^export default ', content, re.MULTILINE)
    if export_match:
        after_export = content[export_match.start():]
        # Look for import statement after the export (sign of duplication)
        import_after = re.search(r'\n(import .+ from .+\n)', after_export[50:])
        if import_after:
            # Content is duplicated, keep only up to where the duplicate starts
            dup_start = export_match.start() + 50 + import_after.start()
            fixed_content = content[:dup_start].rstrip() + '\n'
            print(f'Fixed (import after export): {filepath}')
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            return True

    return False

if __name__ == '__main__':
    frontend_src = os.path.join(os.path.dirname(__file__), 'frontend', 'src')
    fixed_count = 0
    for root, dirs, files in os.walk(frontend_src):
        for file in files:
            if file.endswith('.tsx'):
                filepath = os.path.join(root, file)
                if fix_duplicated_file(filepath):
                    fixed_count += 1

    print(f'Total files fixed: {fixed_count}')
