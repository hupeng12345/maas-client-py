from __future__ import annotations

from enum import StrEnum

PATH_RESULT = "/vision/v1/result"
PATH_PARSE_GENERAL = "/vision/v1/document/parse/general/async"
PATH_PARSE_HUNYUAN = "/vision/v1/document/parse/hunyuan-ocr/async"
PATH_PARSE_MULTILANG = "/vision/v1/document/parse/doc-multilang/async"
PATH_PARSE_ARABIC = "/vision/v1/document/parse/ar-doc/async"
PATH_OCR_GENERAL = "/vision/v1/ocr/general/async"
PATH_OCR_PADDLE = "/vision/v1/ocr/paddle/async"
PATH_OCR_SEAL = "/vision/v1/ocr/general/seals/async"
PATH_OCR_PAGE_DIRECTION = "/vision/v1/ocr/page-direction/async"
PATH_BANK_STATEMENT = "/vision/v1/receipt/bank-statement/async"
PATH_MEDICAL_RECEIPT = "/vision/v1/receipt/medical/async"
PATH_FINANCIAL_RECEIPT = "/vision/v1/receipt/financial/async"
PATH_IMAGE_BLUR = "/vision/v1/ai-detection/image-blur/async"


class CardKind(StrEnum):
    GENERIC = "recognize"
    ID_CARD = "idcard"
    BANK_CARD = "bankcard"
    RESIDENCE_BOOK = "residencebook"
    VEHICLE_LIC = "vehiclelic"
    DRIVER_LIC = "driverlic"
    COMPANY = "company"
    DEGREE = "degree"
    BIRTH = "birth"
    HK_TAIWAN_CARD = "hktaiwancard"
    PASSPORT = "passport"
    MILITARY = "military"
    MARRIAGE = "marriage"

    def path(self, *, sync: bool = True) -> str:
        suffix = "sync" if sync else "async"
        return f"/vision/v1/card/{self.value}/{suffix}"


def card_kinds() -> list[CardKind]:
    return list(CardKind)


# Names mirror the Go constants while retaining Pythonic enum member names.
CardGeneric = CardKind.GENERIC
CardIDCard = CardKind.ID_CARD
CardBankCard = CardKind.BANK_CARD
CardResidenceBook = CardKind.RESIDENCE_BOOK
CardVehicleLic = CardKind.VEHICLE_LIC
CardDriverLic = CardKind.DRIVER_LIC
CardCompany = CardKind.COMPANY
CardDegree = CardKind.DEGREE
CardBirth = CardKind.BIRTH
CardHKTaiwan = CardKind.HK_TAIWAN_CARD
CardPassport = CardKind.PASSPORT
CardMilitary = CardKind.MILITARY
CardMarriage = CardKind.MARRIAGE


__all__ = [
    "PATH_BANK_STATEMENT",
    "PATH_FINANCIAL_RECEIPT",
    "PATH_IMAGE_BLUR",
    "PATH_MEDICAL_RECEIPT",
    "PATH_OCR_GENERAL",
    "PATH_OCR_PADDLE",
    "PATH_OCR_PAGE_DIRECTION",
    "PATH_OCR_SEAL",
    "PATH_PARSE_ARABIC",
    "PATH_PARSE_GENERAL",
    "PATH_PARSE_HUNYUAN",
    "PATH_PARSE_MULTILANG",
    "PATH_RESULT",
    "CardBankCard",
    "CardBirth",
    "CardCompany",
    "CardDegree",
    "CardDriverLic",
    "CardGeneric",
    "CardHKTaiwan",
    "CardIDCard",
    "CardKind",
    "CardMarriage",
    "CardMilitary",
    "CardPassport",
    "CardResidenceBook",
    "CardVehicleLic",
    "card_kinds",
]
