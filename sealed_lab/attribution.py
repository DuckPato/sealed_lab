"""Credit for the 17Lands data, shown in the main window, the HTML export and the README.

The 17Lands public datasets are licensed CC BY 4.0 (credit, link to the license, say what was changed), and the 17Lands
usage guidelines ask tools to name 17Lands with a link, visibly at the top level, without implying endorsement.
"""
from __future__ import annotations

DATASETS_URL = "https://www.17lands.com/public_datasets"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"

CREDIT_TEXT = ("Data: 17Lands public datasets (CC BY 4.0). Win rates, scores and reports are this app's own model of that data; "
               "not affiliated with or endorsed by 17Lands.")


# Wizards of the Coast's Fan Content Policy: free, unofficial fan tools carry this notice.
WOTC_DISCLAIMER = ("sealed_lab is unofficial Fan Content permitted under the Fan Content Policy. Not approved/endorsed by Wizards. "
                   "Portions of the materials used are property of Wizards of the Coast. ©Wizards of the Coast LLC.")
FAN_CONTENT_URL = "https://company.wizards.com/en/legal/fancontentpolicy"


def credit_html(short: bool = False) -> str:
    """The credit as HTML with links.  `short`: one compact line (the main window's tab bar)."""
    src = f'<a href="{DATASETS_URL}">17Lands public datasets</a>'
    lic = f'<a href="{LICENSE_URL}">CC BY 4.0</a>'
    if short:
        return f"Data: {src} ({lic}) · not affiliated with 17Lands"
    return (f"Data: {src}, licensed {lic}. Win rates, scores and reports are this app's own model of that data; "
            "not affiliated with or endorsed by 17Lands.")
