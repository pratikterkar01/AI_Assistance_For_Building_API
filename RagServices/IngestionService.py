from pathlib import Path
from docling_core.types.doc import DoclingDocument
from Utils import Config,logger
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions,RapidOcrOptions
from docling.datamodel.accelerator_options import AcceleratorDevice,AcceleratorOptions
logger = logger.get_logger(__name__)
global_variable = Config.get_settings()
class Ingestion:

    @staticmethod
    def parse_pdf(pdf_path: str) -> DoclingDocument:
        """Parse PDF using Docling with NVIDIA GPU acceleration."""
        pipeline_options = PdfPipelineOptions()
        pipeline_options.ocr_options = RapidOcrOptions(backend="torch")
        pipeline_options.accelerator_options = AcceleratorOptions(
            device=AcceleratorDevice.CUDA,  # or AUTO
            num_threads=8,
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(pdf_path)
        return result.document


    ## save in the text format
    @staticmethod
    def save_parsed_document(pdf_path: str, parsed_dir: Path = global_variable.PARSED_DIR) -> Path:
        try:
            logger.info(f"Saving parsed document start for: {pdf_path}")
            parsed_dir = Path(parsed_dir)
            parsed_dir.mkdir(parents=True, exist_ok=True)
            txt_path = parsed_dir / f"{Path(pdf_path).stem}.txt"

            if txt_path.exists():
                print(f"Already parsed, skipping: {txt_path.name}")
                return txt_path

            doc = Ingestion.parse_pdf(pdf_path)
            txt_path.write_text(doc.model_dump_json(), encoding="utf-8")
            logger.info(f"Saved parsed document: {txt_path}")
            return txt_path
        except Exception as e:
            logger.error(f"Error while saving parsed document: {e}")
            return ""

    @staticmethod
    def load_parsed_document(txt_path: Path) -> DoclingDocument:
        text = Path(txt_path).read_text(encoding="utf-8")
        return DoclingDocument.model_validate_json(text)