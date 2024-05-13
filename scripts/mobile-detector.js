let isVertical = false;

window.addEventListener("resize", checkAspect);
window.addEventListener("load", checkAspect);

function checkAspect() {
	const oldIsVertical = isVertical;

	const ratio = window.innerWidth / window.innerHeight;
	isVertical = ratio < 1.35;

	if (isVertical != oldIsVertical) {
		document.dispatchEvent(new Event("switchRatio"));

		if (isVertical) {
			document.getRootNode().childNodes[1].childNodes[0].innerHTML += `<link rel="stylesheet" href="mobile-style.css" id="_mobile-style-override">`
		} else {
			const override = document.getElementById("_mobile-style-override");

			if (override != null) override.remove();
		}
	}
}