import os
import re
from typing import List

import PIL
from PIL.Image import Image



def clean_string(s):
    """
    Remove non-alphanumeric characters except spaces, and normalize spacing.
    Args:
        s: The string to clean.

    Returns: A cleaned string.
    """
    # Remove non-alphanumeric characters except spaces
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    # Check for a sentence with just the same word repeated
    if len(set(cleaned.split())) == 1:
        cleaned = cleaned.split()[0]
    # Replace multiple spaces with a single space
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


class ImageData:
    image_path: str = ""
    caption: str = ""
    tags: List[str] = []
    selected: bool = False
    filtered: bool = False
    image = None
    id = None

    def __init__(self, image_path):
        self.image_path = image_path
        self.caption = self.read_caption()
        self.tags = self.split_caption()
        # Generate a random id
        self.id = os.urandom(32).hex()

    def read_caption(self):
        existing_caption_txt_filename = os.path.splitext(self.image_path)[0] + '.txt'
        if os.path.exists(existing_caption_txt_filename):
            with open(existing_caption_txt_filename, 'r', encoding="utf8") as file:
                existing_caption_txt = file.read()
                existing_caption_txt = existing_caption_txt.strip()
        else:
            image_name = os.path.splitext(os.path.basename(self.image_path))[0]
            existing_caption_txt = clean_string(image_name)
        return existing_caption_txt

    def split_caption(self):
        tags = self.caption.split(",")
        tags = [tag.strip() for tag in tags]
        tags = [tag for tag in tags if tag != ""]
        return tags

    def update_image(self, image: Image, save_file: bool = False):
        if save_file:
            img_path = os.path.splitext(self.image_path)[0] + '.png'
            if img_path != self.image_path and os.path.exists(self.image_path):
                os.remove(self.image_path)
            self.image_path = img_path
            image.save(self.image_path)
        self.image = image

    def update_caption(self, caption: str, save_file: bool = False):
        if save_file:
            caption_txt_filename = os.path.splitext(self.image_path)[0] + '.txt'
            with open(caption_txt_filename, 'w', encoding="utf8") as file:
                file.write(caption)
        self.caption = caption
        self.tags = self.split_caption()

    def get_image(self):
        if self.image is None:
            self.image = PIL.Image.open(self.image_path).convert("RGB")
        return self.image


class FileManager:
    file_path: str = ""
    _instance = None
    files: List[ImageData] = []
    included_tags: List[str] = []
    excluded_tags: List[str] = []
    included_strings: List[str] = []
    excluded_strings: List[str] = []
    current_image = None

    def __init__(self):
        self.files = []

    def __new__(cls):
        if FileManager._instance is None:
            FileManager._instance = object.__new__(cls)
        return FileManager._instance

    def clear(self):
        self.files = []

    def load_files(self):
        from extensions.sd_smartprocess.smartprocess import is_image

        self.clear()
        # Walk through all files in the directory that contains the images
        for root, dirs, files in os.walk(self.file_path):
            for file in files:
                file = os.path.join(root, file)
                if is_image(file) and "_backup" not in file:
                    image_data = ImageData(file)
                    self.files.append(image_data)
        self.update_filters()

    def filtered_files(self, for_gallery: bool = False):
        if not for_gallery:
            return [file for file in self.files if file.filtered]
        else:
            return [(file.image_path, file.caption) for file in self.files if file.filtered]

    def all_files(self, for_gallery: bool = False) -> List[ImageData]:
        if not for_gallery:
            return self.files
        else:
            return [(file.image_path, file.caption) for file in self.files]

    def selected_files(self, for_gallery: bool = False) -> List[ImageData]:
        if not for_gallery:
            return [file for file in self.files if file.selected]
        else:
            return [(file.image_path, file.caption) for file in self.files if file.selected]

    def filtered_and_selected_files(self, for_gallery: bool = False) -> List[ImageData]:
        if not for_gallery:
            return [file for file in self.files if file.filtered and file.selected and file.filtered]
        else:
            return [(file.image_path, file.caption) for file in self.files if file.selected and file.filtered]

    def update_files(self, files: List[ImageData]):
        for file in files:
            self.update_file(file)

    def update_file(self, file: ImageData):
        # Search for the file with the same ID and update it if found
        for i, existing_file in enumerate(self.files):
            if existing_file.id == file.id:
                self.files[i] = file
                break
        else:
            # The file was not found in the list, append it
            self.files.append(file)

    def update_filters(self):
        """
        Filters a collection of files based on specified inclusion and exclusion criteria for tags and filter strings.

        The function filters files based on tags extracted from file captions and matches these tags against specified
        inclusion and exclusion criteria. The criteria can be a set of plain tags, wildcard patterns, or regex expressions.

        Parameters:
        - use_all_files (bool): Determines whether to filter from all files or only selected files.
          If True, filters from all files; otherwise, filters from selected files.

        Globals:
        - filter_tags_include (list): Tags required to include a file.
        - filter_tags_exclude (list): Tags that lead to exclusion of a file.
        - filter_string_include (list): Filter strings for inclusion; supports wildcards and regex.
        - filter_string_exclude (list): Filter strings for exclusion; supports wildcards and regex.
        - all_files (list): List of all files, where each file is a tuple (filename, caption).
        - selected_files (list): List of selected files, where each file is a tuple (filename, caption).

        Returns:
        - filtered_files (list): List of files filtered based on the specified criteria.

        Examples:
        1. Plain Tag Matching:
           - To include files with a tag 'holiday', add 'holiday' to filter_tags_include.
           - To exclude files with a tag 'work', add 'work' to filter_tags_exclude.

        2. Wildcard Patterns:
           - To include files with tags starting with 'trip', add 'trip*' to filter_string_include.
           - To exclude files with tags ending with '2023', add '*2023' to filter_string_exclude.

        3. Regex Expressions:
           - To include files with tags that have any number followed by 'days', add '\\d+days' to filter_string_include.
           - To exclude files with tags formatted like dates (e.g., '2023-04-01'), add '\\d{4}-\\d{2}-\\d{2}' to filter_string_exclude.

        Note:
        - The function treats tags as case-sensitive.
        - Wildcard '*' matches any sequence of characters (including none).
        - Regex patterns should follow Python's 're' module syntax.
        """

        def matches_pattern(pattern, string):
            # Convert wildcard to regex if necessary
            if '*' in pattern:
                pattern = '^' + pattern.replace('*', '.*') + '$'
                return re.match(pattern, string) is not None
            else:
                if " " not in pattern:
                    parts = string.split(" ")
                    return pattern in parts
                else:
                    return pattern in string

        def should_include(tag, filter_tags, filter_strings):
            if len(filter_tags) == 0 and len(filter_strings) == 0:
                return True
            tag_match = False
            filter_match = False
            if tag in filter_tags and len(filter_tags) > 0:
                tag_match = True
            if len(filter_strings) > 0:
                for filter_string in filter_strings:
                    if matches_pattern(filter_string, tag):
                        filter_match = True
                        break
            return tag_match or filter_match

        def should_exclude(tag, filter_tags, filter_strings):
            if tag in filter_tags:
                return True
            for filter_string in filter_strings:
                if matches_pattern(filter_string, tag):
                    return True
            return False

        files = self.files
        updated_files = []
        for file in files:
            tags = file.tags
            out_tags = []
            for tag in tags:
                include = True
                if should_exclude(tag, self.excluded_tags, self.excluded_strings):
                    include = False
                elif not should_include(tag, self.included_tags, self.included_strings):
                    include = False
                if include:
                    out_tags.append(tag)
            file.filtered = len(out_tags) > 0
            updated_files.append(file)
        self.files = files

    def all_captions(self):
        return [file.caption for file in self.files]
