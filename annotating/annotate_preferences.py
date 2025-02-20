import sys
import pandas as pd
import json
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QTextEdit, QFileDialog, QMessageBox, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon, QPixmap, QPalette, QBrush

class AnnotationApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.set_background_image()

        self.setWindowTitle("LLM Response Annotation Tool")
        self.setGeometry(100, 100, 1400, 900)  # Increased width to accommodate new annotation box

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # File loading section
        file_layout = QHBoxLayout()
        self.load_button = QPushButton("Load CSV")
        self.load_button.clicked.connect(self.load_csv)
        file_layout.addWidget(self.load_button)
        
        self.save_button = QPushButton("Save Annotations")
        self.save_button.clicked.connect(self.save_annotations)
        self.save_button.setEnabled(False)
        file_layout.addWidget(self.save_button)
        
        main_layout.addLayout(file_layout)

        # Main content layout
        content_layout = QHBoxLayout()
        
        # Left column for prompt and annotations
        left_column = QVBoxLayout()
        left_column.setSpacing(20)
        
        # Prompt display
        prompt_label = QLabel("Prompt:")
        prompt_label.setFont(QFont("Arial", 12, QFont.Bold))
        left_column.addWidget(prompt_label)
        self.prompt_text = QTextEdit()
        self.prompt_text.setReadOnly(True)
        self.prompt_text.setFont(QFont("Arial", 11))
        left_column.addWidget(self.prompt_text)

        # Target Annotations display
        target_annotations_label = QLabel("Target Annotations:")
        target_annotations_label.setFont(QFont("Arial", 12, QFont.Bold))
        left_column.addWidget(target_annotations_label)
        self.target_annotations_text = QTextEdit()
        self.target_annotations_text.setReadOnly(True)
        self.target_annotations_text.setFont(QFont("Arial", 11))
        left_column.addWidget(self.target_annotations_text)

        # Target File Path display
        target_file_path_label = QLabel("Target File Path:")
        target_file_path_label.setFont(QFont("Arial", 12, QFont.Bold))
        left_column.addWidget(target_file_path_label)
        self.target_file_path_text = QTextEdit()
        self.target_file_path_text.setReadOnly(True)
        self.target_file_path_text.setFont(QFont("Arial", 11))
        left_column.addWidget(self.target_file_path_text)

        # Right column for responses
        right_column = QVBoxLayout()
        right_column.setSpacing(20)
        
        # Response 1 section
        response1_layout = QVBoxLayout()
        response1_label = QLabel("Model Response 1")
        response1_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.response1_button = QPushButton("Select Response 1")
        self.response1_button.setCheckable(True)
        self.response1_button.clicked.connect(lambda: self.select_response(1))
        
        self.response1_text = QTextEdit()
        self.response1_text.setReadOnly(True)
        self.response1_text.setFont(QFont("Arial", 11))
        
        response1_layout.addWidget(response1_label)
        response1_layout.addWidget(self.response1_text)
        response1_layout.addWidget(self.response1_button)
        
        # Response 2 section
        response2_layout = QVBoxLayout()
        response2_label = QLabel("Model Response 2")
        response2_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.response2_button = QPushButton("Select Response 2")
        self.response2_button.setCheckable(True)
        self.response2_button.clicked.connect(lambda: self.select_response(2))
        
        self.response2_text = QTextEdit()
        self.response2_text.setReadOnly(True)
        self.response2_text.setFont(QFont("Arial", 11))
        
        response2_layout.addWidget(response2_label)
        response2_layout.addWidget(self.response2_text)
        response2_layout.addWidget(self.response2_button)

        # Add response layouts to right column
        right_column.addLayout(response1_layout)
        right_column.addLayout(response2_layout)

        # Combine left and right columns
        content_layout.addLayout(left_column, 1)  # 1 is the stretch factor
        content_layout.addLayout(right_column, 2)
        
        main_layout.addLayout(content_layout)

        # Navigation buttons
        nav_layout = QHBoxLayout()
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_row)
        self.next_button.setEnabled(False)
        nav_layout.addWidget(self.next_button)
        
        main_layout.addLayout(nav_layout)

        # Initialize tracking variables
        self.current_df = None
        self.current_index = 0
        self.annotations = []

    def load_csv(self):
        """Load CSV file and prepare for annotation"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Annotation CSV", "", "CSV Files (*.csv)")
        if file_path:
            try:
                self.current_df = pd.read_csv(file_path)
                self.current_index = 0
                self.annotations = [None] * len(self.current_df)
                self.update_display()
                
                self.next_button.setEnabled(True)
                self.save_button.setEnabled(True)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load CSV: {str(e)}")

    def update_display(self):
        """Update the display with the current row's data"""
        if self.current_df is not None and self.current_index < len(self.current_df):
            row = self.current_df.iloc[self.current_index]
            
            # Set prompt
            self.prompt_text.setText(str(row['Prompt']))
            
            # Set target annotations (parse JSON if applicable)
            try:
                target_annotations = row['Target Annotations']
                if isinstance(target_annotations, str):
                    # Try to parse JSON and pretty print
                    parsed_annotations = json.loads(target_annotations)
                    self.target_annotations_text.setText(json.dumps(parsed_annotations, indent=2))
                else:
                    self.target_annotations_text.setText(str(target_annotations))
            except Exception as e:
                self.target_annotations_text.setText(f"Error parsing annotations: {str(e)}")
            
            # Set model responses
            self.response1_text.setText(str(row['Model Response 1']))
            self.response2_text.setText(str(row['Model Response 2']))

            # Set target file path
            self.target_file_path_text.setText(str(row.get('Target File Path', 'No path provided')))
            
            # Reset button states based on current annotation selection
            self.response1_button.setChecked(self.annotations[self.current_index] == 1)
            self.response2_button.setChecked(self.annotations[self.current_index] == 2)

    def select_response(self, response_num):
        """Record the selected response"""
        if self.current_df is not None:
            self.annotations[self.current_index] = response_num
            self.update_display()

    def next_row(self):
        """Move to the next row"""
        if self.current_df is not None:
            if self.current_index < len(self.current_df) - 1:
                self.current_index += 1
                self.update_display()
            else:
                QMessageBox.information(self, "End", "You've reached the last row!")
                self.next_button.setEnabled(False)  # Disable 'Next' button when at the last row

    def save_annotations(self):
        """Save annotations to a new CSV"""
        if self.current_df is not None:
            # Ensure all rows have been annotated
            if None in self.annotations:
                reply = QMessageBox.question(self, 'Incomplete Annotations', 
                                             'Some rows are not annotated. Save anyway?', 
                                             QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.No:
                    return

            # Create a copy of the dataframe to modify
            annotated_df = self.current_df.copy()
            
            # Add annotation column
            annotated_df['Preferred_Response'] = self.annotations
            
            # Prompt for save location
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Annotated CSV", "", "CSV Files (*.csv)")
            
            if save_path:
                try:
                    annotated_df.to_csv(save_path, index=False)
                    QMessageBox.information(self, "Success", f"Annotations saved to {save_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Could not save CSV: {str(e)}")

    def set_background_image(self):
        """Set a background image for the application"""
        # You would replace 'background.jpg' with your actual background image path
        background_path = self.get_resource_path('background.jpg')
        if os.path.exists(background_path):
            palette = QPalette()
            pixmap = QPixmap(background_path)
            brush = QBrush(pixmap)
            palette.setBrush(QPalette.Background, brush)
            self.setPalette(palette)
            self.setAutoFillBackground(True)

    def get_resource_path(self, relative_path):
        """
        Get absolute path to resource file. Works for dev and for PyInstaller
        """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)
    

def main():
    app = QApplication(sys.argv)
    annotation_app = AnnotationApp()
    annotation_app.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
