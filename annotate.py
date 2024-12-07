import sys
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QTextEdit, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt

class AnnotationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM Response Annotation Tool")
        self.setGeometry(100, 100, 1200, 800)

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

        # Prompt display
        prompt_label = QLabel("Prompt:")
        main_layout.addWidget(prompt_label)
        self.prompt_text = QTextEdit()
        self.prompt_text.setReadOnly(True)
        main_layout.addWidget(self.prompt_text)

        # Side-by-side response layout
        response_layout = QHBoxLayout()
        
        # Response 1 section
        response1_layout = QVBoxLayout()
        response1_label = QLabel("Model Response 1")
        response1_button = QPushButton("Select Response 1")
        response1_button.clicked.connect(lambda: self.select_response(1))
        
        self.response1_text = QTextEdit()
        self.response1_text.setReadOnly(True)
        
        response1_layout.addWidget(response1_label)
        response1_layout.addWidget(self.response1_text)
        response1_layout.addWidget(response1_button)
        
        # Response 2 section
        response2_layout = QVBoxLayout()
        response2_label = QLabel("Model Response 2")
        response2_button = QPushButton("Select Response 2")
        response2_button.clicked.connect(lambda: self.select_response(2))
        
        self.response2_text = QTextEdit()
        self.response2_text.setReadOnly(True)
        
        response2_layout.addWidget(response2_label)
        response2_layout.addWidget(self.response2_text)
        response2_layout.addWidget(response2_button)

        # Add response layouts to main response layout
        response_layout.addLayout(response1_layout)
        response_layout.addLayout(response2_layout)
        
        main_layout.addLayout(response_layout)

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
        """Update the display with current row's data"""
        if self.current_df is not None:
            row = self.current_df.iloc[self.current_index]
            
            # Set prompt
            self.prompt_text.setText(str(row['Prompt']))
            
            # Set model responses
            self.response1_text.setText(str(row['Model Response 1']))
            self.response2_text.setText(str(row['Model Response 2']))

    def select_response(self, response_num):
        """Record the selected response"""
        if self.current_df is not None:
            self.annotations[self.current_index] = response_num
            QMessageBox.information(self, "Selection", f"Response {response_num} selected")

    def next_row(self):
        """Move to next row"""
        if self.current_df is not None:
            if self.current_index < len(self.current_df) - 1:
                self.current_index += 1
                self.update_display()
            else:
                QMessageBox.information(self, "End", "You've reached the last row!")

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

def main():
    app = QApplication(sys.argv)
    annotation_app = AnnotationApp()
    annotation_app.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()