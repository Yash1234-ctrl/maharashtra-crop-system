import sys
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.debug(f"Python version: {sys.version}")
        logger.debug(f"Streamlit version: {st.__version__}")
        logger.debug("Starting Streamlit application")
        
        st.title("Debug Test")
        st.write("If you can see this, Streamlit is working!")
        
        logger.debug("Application started successfully")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    main()