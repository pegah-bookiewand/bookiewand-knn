from enum import Enum
from dataclasses import dataclass
from typing import Optional


@dataclass
class TaxTypeInfo:
    """Information about a specific tax type"""
    rate: float
    name: str


class TaxType(str, Enum):
    """
    Enumeration of tax types with their rates and descriptive names.
    
    This can be used for consistent tax type identification throughout the application.
    """
    OUTPUT = "OUTPUT"               # GST on Income (10%)
    INPUT = "INPUT"                 # GST on Expenses (10%)
    EXEMPTEXPENSES = "EXEMPTEXPENSES"  # GST Free Expenses (0%)
    EXEMPTOUTPUT = "EXEMPTOUTPUT"   # GST Free Income (0%)
    BASEXCLUDED = "BASEXCLUDED"     # BAS Excluded (0%)
    GSTONIMPORTS = "GSTONIMPORTS"   # GST on Imports (0%)
    
    # Define this as a class variable to avoid any potential issues with enum methods
    _tax_info = {
        OUTPUT: TaxTypeInfo(rate=10.00, name="GST on Income"),
        INPUT: TaxTypeInfo(rate=10.00, name="GST on Expenses"),
        EXEMPTEXPENSES: TaxTypeInfo(rate=0.00, name="GST Free Expenses", is_reportable=True),
        EXEMPTOUTPUT: TaxTypeInfo(rate=0.00, name="GST Free Income"),
        BASEXCLUDED: TaxTypeInfo(rate=0.00, name="BAS Excluded", is_reportable=False),
        GSTONIMPORTS: TaxTypeInfo(rate=0.00, name="GST on Imports")
    }
    
    @property
    def tax_rate(self) -> float:
        """Get the tax rate for this tax type"""
        return self._tax_info[self].rate
    
    @property
    def tax_name(self) -> str:
        """Get the descriptive name for this tax type"""
        return self._tax_info[self].name
    
    @classmethod
    def from_name(cls, name: str) -> Optional['TaxType']:
        """Find a tax type by its descriptive name"""
        for tax_type, info in cls._tax_info.items():
            if info.name.lower() == name.lower():
                return tax_type
        return None
    
    @classmethod
    def from_rate(cls, rate: float, is_input: bool = False) -> Optional['TaxType']:
        """
        Find a tax type by its rate and whether it's for input (expenses) or output (income)
        
        Args:
            rate: The tax rate to search for
            is_input: True if this is for expenses, False for income
            
        Returns:
            The matching TaxType or None if no match found
        """
        if rate == 10.0:
            return cls.INPUT if is_input else cls.OUTPUT
        elif rate == 0.0:
            return cls.EXEMPTEXPENSES if is_input else cls.EXEMPTOUTPUT
        return None 