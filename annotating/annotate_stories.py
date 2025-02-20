import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import openpyxl  # for writing Excel
import os
import re
import sys

###############################################################################
# CONFIGURATION
###############################################################################
# We no longer hardcode an Excel path. Instead, we'll prompt the user to select it.

# Excel columns for the relevant data (these must match your actual .xlsx file)
TARGET_ANNOTATION_COL = "Target File Path"
GROUP_ID_COL         = "GroupID"
STORY_ORDER_COL      = "StoryOrder"
EXTRACTED_STORY_COL  = "Extracted Story"
MODEL_RESPONSE_COL   = "Model Response 1"

# Per-story question columns
STORY_Q1 = "Story Q1: Is the story accurate? (Yes/No)"
STORY_Q2 = "Story Q2: If yes, could the story use more precise behaviors? (Yes/No)"
STORY_Q3 = "Story Q3: If yes, please provide the more precise behaviors it should.."
STORY_Q4 = "Story Q4: If applicable, please rewrite the story to be as precise as possible."

# Overall annotation question (once per group)
OVERALL_Q = "Please provide any stories which this output is missing"

###############################################################################
# HELPER FUNCTIONS
###############################################################################
def split_annotation(file_text):
    """
    Finds the first occurrence of '(A:' in the file_text.
    Everything BEFORE that marker goes to the main_text (top box).
    Everything FROM '(A:' to the end goes to the annotation_text (bottom box).

    If '(A:' is not found, we return the entire file_text as main_text
    and an empty string as annotation_text.
    """
    marker = "(A:"
    idx = file_text.find(marker)
    if idx == -1:
        return file_text.strip(), ""
    main_text = file_text[:idx].strip()
    annotation_text = file_text[idx:].strip()
    return main_text, annotation_text

def format_s_sections(annotation_text):
    """
    In the annotation_text, find all segments that begin with '(S:' and end with ')'.
    For each such segment, split the content on '.' and insert newlines so each
    sentence is on its own line.
    """
    pattern = r"\(S:(.*?)\)"

    def repl(match):
        inner = match.group(1).strip()
        replaced = inner.replace('.', '.\n')
        return f"(S: {replaced})"

    return re.sub(pattern, repl, annotation_text, flags=re.DOTALL)

###############################################################################
# MAIN APPLICATION CLASS
###############################################################################
class AnnotationApp(tk.Tk):
    """
    A Tkinter GUI that displays (page by page) the groups of stories
    from an expanded Excel file. The left side is split vertically:
      - Top (85%): main text from file
      - Bottom (15%): 'Annotations' box
    The right side is scrollable for stories, overall Q, model response, and buttons.

    1) We split the file at '(A:' so everything from that marker downward goes
       into the annotations box.
    2) Within that annotations text, we look for '(S: ... )' blocks and insert
       newlines after each '.' for readability.
    """
    def __init__(self, excel_path):
        super().__init__()
        self.title("Annotation App")
        self.geometry("1200x800")

        # 1) Load the Excel data into a DataFrame
        self.excel_path = excel_path
        self.df = pd.read_excel(excel_path, engine="openpyxl")

        # 2) Identify unique groups
        self.group_ids = sorted(self.df[GROUP_ID_COL].unique())
        self.current_group_index = 0

        # 3) Build the main UI layout
        self._build_ui()

        # 4) Load the first page (if any)
        if len(self.group_ids) == 0:
            messagebox.showinfo("No Data", "No groups found in the Excel file.")
        else:
            self.load_page(self.current_group_index)

    def _build_ui(self):
        """
        Builds a 2-column layout:
          - Left column: top (85%) for main text, bottom (15%) for annotation text
          - Right column: a scrollable frame for stories, overall Q, model response, and buttons
        """
        # Two main columns
        self.columnconfigure(0, weight=1, uniform="col")
        self.columnconfigure(1, weight=1, uniform="col")
        self.rowconfigure(0, weight=1)

        # ------------------
        # LEFT SIDE
        # ------------------
        self.left_frame = tk.Frame(self, bd=2, relief="groove")
        self.left_frame.grid(row=0, column=0, sticky="nsew")

        # We'll split left_frame into two rows: top (85%) and bottom (15%)
        self.left_frame.rowconfigure(0, weight=85)
        self.left_frame.rowconfigure(1, weight=15)
        self.left_frame.columnconfigure(0, weight=1)

        # -- TOP (for main text) --
        top_left_frame = tk.Frame(self.left_frame)
        top_left_frame.grid(row=0, column=0, sticky="nsew")

        label_txt = tk.Label(top_left_frame, text="Target Annotation File Content:", font=("Arial", 12, "bold"))
        label_txt.pack(anchor="nw", padx=5, pady=5)

        self.text_file_box = tk.Text(top_left_frame, wrap="word")
        self.text_file_box.pack(fill="both", expand=True, padx=5, pady=5)
        self.text_file_box.configure(state="disabled")

        # -- BOTTOM (for annotation text) --
        bottom_left_frame = tk.Frame(self.left_frame)
        bottom_left_frame.grid(row=1, column=0, sticky="nsew")

        annotation_label = tk.Label(bottom_left_frame, text="Annotations", font=("Arial", 10, "bold"))
        annotation_label.pack(anchor="nw", padx=5, pady=2)

        self.annotation_box = tk.Text(bottom_left_frame, wrap="word", height=4)
        self.annotation_box.pack(fill="both", expand=True, padx=5, pady=5)
        self.annotation_box.configure(state="disabled")

        # ------------------
        # RIGHT SIDE
        # ------------------
        self.right_frame = tk.Frame(self, bd=2, relief="groove")
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        # We'll make a scrollable canvas in right_frame
        self.canvas = tk.Canvas(self.right_frame)
        self.canvas.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(self.right_frame, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side="right", fill="y")

        self.canvas.configure(yscrollcommand=scrollbar.set)

        # A frame inside the canvas, which will hold all our content
        self.scrollable_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.scrollable_frame.bind("<Configure>", self._on_scrollable_frame_configure)

        # Inside scrollable_frame, sub-sections:
        self.stories_frame = tk.Frame(self.scrollable_frame)
        self.stories_frame.pack(fill="x", padx=5, pady=5)
        self.story_widgets = []

        self.overall_label = tk.Label(self.scrollable_frame, text=OVERALL_Q, font=("Arial", 10, "bold"))
        self.overall_label.pack(anchor="w", padx=5, pady=2)

        self.overall_stories_label = tk.Label(self.scrollable_frame, text="", font=("Arial", 10), justify="left")
        self.overall_stories_label.pack(anchor="w", padx=5, pady=2)

        self.overall_text = tk.Text(self.scrollable_frame, height=3, wrap="word")
        self.overall_text.pack(fill="x", padx=5, pady=5)

        tk.Label(self.scrollable_frame, text="Model Response:", font=("Arial", 12, "bold")).pack(
            anchor="w", padx=5, pady=5
        )
        self.model_response_box = tk.Text(self.scrollable_frame, height=8, wrap="word")
        self.model_response_box.pack(fill="x", padx=5, pady=5)

        btn_frame = tk.Frame(self.scrollable_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)

        self.save_button = tk.Button(btn_frame, text="Save", command=self.save_page)
        self.save_button.pack(side="left", padx=5, pady=5, expand=True, fill="x")

        self.next_button = tk.Button(btn_frame, text="Next", command=self.next_page)
        self.next_button.pack(side="left", padx=5, pady=5, expand=True, fill="x")

    def _on_scrollable_frame_configure(self, event):
        """
        Reset the scroll region to encompass the entire scrollable_frame.
        """
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def load_page(self, group_index):
        """
        Loads the specified group's data into the UI:
          - Reads the 'Target File Path' from the first row in this group,
            opens that file, then splits it at '(A:' so everything from that
            marker onward is displayed in the bottom annotation box, and
            everything before it is in the top text box.
          - Then, in the annotation text, for any '(S: ... )' block, we
            split sentences by '.' and insert newlines for readability.
          - Creates sub-frames for each story in that group (with Q1–Q4).
          - Lists all stories under the overall question label, then
            shows the overall question text from the first row.
          - Shows the model response text from the first row.
        """
        # Clear old story frames
        for child in self.stories_frame.winfo_children():
            child.destroy()
        self.story_widgets.clear()

        if group_index < 0 or group_index >= len(self.group_ids):
            return

        group_id = self.group_ids[group_index]
        group_df = self.df[self.df[GROUP_ID_COL] == group_id]

        # 1) Load the target annotation file from the first row
        first_row = group_df.iloc[0]
        annotation_path = "../" + str(first_row.get(TARGET_ANNOTATION_COL, ""))

        file_text = self._read_text_file(annotation_path)
        
        # 2) Split out everything below (A:
        main_text, ann_text = split_annotation(file_text)

        # 3) Within the annotation text, find (S: ... ) blocks and insert newlines after each period
        ann_text = format_s_sections(ann_text)

        # Display main text in the top text_file_box
        self.text_file_box.configure(state="normal")
        self.text_file_box.delete("1.0", "end")
        self.text_file_box.insert("1.0", main_text)
        self.text_file_box.configure(state="disabled")

        # Display annotations in the bottom annotation_box
        self.annotation_box.configure(state="normal")
        self.annotation_box.delete("1.0", "end")
        self.annotation_box.insert("1.0", ann_text)
        self.annotation_box.configure(state="disabled")

        # 4) Build sub-frames for each story in the group (on the right side)
        for row_idx, row_data in group_df.iterrows():
            story_text = str(row_data.get(EXTRACTED_STORY_COL, ""))

            frame = tk.Frame(self.stories_frame, borderwidth=1, relief="groove")
            frame.pack(fill="x", pady=5)

            # Story label
            tk.Label(frame, text=f"Story: {story_text}", font=("Arial", 10, "bold")).pack(anchor="w", padx=5, pady=2)

            # Q1 dropdown
            q1_val = tk.StringVar(value=row_data.get(STORY_Q1, "null"))
            tk.Label(frame, text=STORY_Q1).pack(anchor="w", padx=20, pady=2)
            q1_combo = ttk.Combobox(frame, values=["null", "Yes", "No"], textvariable=q1_val, state="readonly")
            q1_combo.pack(anchor="w", padx=40, pady=2)

            # Q2 dropdown
            q2_val = tk.StringVar(value=row_data.get(STORY_Q2, "null"))
            tk.Label(frame, text=STORY_Q2).pack(anchor="w", padx=20, pady=2)
            q2_combo = ttk.Combobox(frame, values=["null", "Yes", "No"], textvariable=q2_val, state="readonly")
            q2_combo.pack(anchor="w", padx=40, pady=2)

            # Q3 text
            tk.Label(frame, text=STORY_Q3).pack(anchor="w", padx=20, pady=2)
            q3_text = tk.Text(frame, height=2, wrap="word")
            q3_text.pack(anchor="w", padx=40, pady=2, fill="x")
            q3_text.insert("1.0", str(row_data.get(STORY_Q3, "") or ""))

            # Q4 text
            tk.Label(frame, text=STORY_Q4).pack(anchor="w", padx=20, pady=2)
            q4_text = tk.Text(frame, height=2, wrap="word")
            q4_text.pack(anchor="w", padx=40, pady=2, fill="x")
            q4_text.insert("1.0", str(row_data.get(STORY_Q4, "") or ""))

            # Save references for saving
            self.story_widgets.append({
                "row_index": row_idx,
                "q1_var": q1_val,
                "q2_var": q2_val,
                "q3_text": q3_text,
                "q4_text": q4_text
            })

        # 5) Update the label that lists all stories in this group
        group_stories = []
        for _, row_data in group_df.iterrows():
            group_stories.append(str(row_data.get(EXTRACTED_STORY_COL, "")))
        stories_string = "\n-----\n".join(group_stories)
        self.overall_stories_label.config(text=stories_string)

        # 6) Overall question
        overall_val = str(first_row.get(OVERALL_Q, "") or "")
        self.overall_text.delete("1.0", "end")
        self.overall_text.insert("1.0", overall_val)

        # 7) Model response
        model_response = str(first_row.get(MODEL_RESPONSE_COL, "") or "")
        self.model_response_box.delete("1.0", "end")
        self.model_response_box.insert("1.0", model_response)

        # Reset the scroll position
        self.canvas.yview_moveto(0)

    def _read_text_file(self, path):
        """
        Safely read the text file at 'path'. If it doesn't exist, return an error message.
        """
        if not path or not os.path.exists(path):
            return f"(No file or file not found: {path})"
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file {path}:\n{e}"

    def save_page(self):
        """
        Saves the user’s inputs for the current group back into the DataFrame,
        then writes the entire DataFrame to the Excel file.
        """
        if self.current_group_index < 0 or self.current_group_index >= len(self.group_ids):
            return

        group_id = self.group_ids[self.current_group_index]
        group_df = self.df[self.df[GROUP_ID_COL] == group_id]

        # 1) Save story-level answers
        for widget_info in self.story_widgets:
            row_idx = widget_info["row_index"]
            self.df.at[row_idx, STORY_Q1] = widget_info["q1_var"].get()
            self.df.at[row_idx, STORY_Q2] = widget_info["q2_var"].get()
            q3_val = widget_info["q3_text"].get("1.0", "end").strip()
            q4_val = widget_info["q4_text"].get("1.0", "end").strip()
            self.df.at[row_idx, STORY_Q3] = q3_val
            self.df.at[row_idx, STORY_Q4] = q4_val

        # 2) Save overall question (once per group, but stored in all rows for simplicity)
        overall_val = self.overall_text.get("1.0", "end").strip()
        for row_idx in group_df.index:
            self.df.at[row_idx, OVERALL_Q] = overall_val

        # 3) Write updated DataFrame to Excel
        try:
            self.df.to_excel(self.excel_path, index=False, engine="openpyxl")
            messagebox.showinfo("Saved", "Your answers have been saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to write to Excel:\n{e}")

    def next_page(self):
        """
        Saves the current page, then moves to the next page if available.
        """
        self.save_page()
        if self.current_group_index < len(self.group_ids) - 1:
            self.current_group_index += 1
            self.load_page(self.current_group_index)
        else:
            messagebox.showinfo("Done", "No more pages.")


###############################################################################
# MAIN SCRIPT
###############################################################################
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the extra root window while we pick a file

    excel_path = filedialog.askopenfilename(
        title="Select the Excel file to annotate",
        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
    )
    if not excel_path:
        messagebox.showinfo("No File Selected", "No Excel file was selected. Exiting.")
        sys.exit(0)

    root.destroy()  # Destroy the file dialog root

    app = AnnotationApp(excel_path)
    app.mainloop()
