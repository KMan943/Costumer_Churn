import sys

from logger import logging

def error_message_detail(error , error_detail:sys):
    _,_,error_tb = error_detail.exc_info()
    file_name = error_tb.tb_frame.f_code.co_filename
    
    error_message = f"Error occured in python script: '{file_name}'\n at line number: '{error_tb.tb_lineno}'\n error message: {str(error)}"
    
    return error_message

class CustomExeption(Exception):
    def __init__(self,error_message , error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message , error_detail)

    def __str__(self):
        return self.error_message
    