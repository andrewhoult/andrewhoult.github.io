// Wow! This is so unnecessary!

function getYearName(academicYear) {
	switch (academicYear) {
		case 0:
			return "first";
		case 1:
			return "second";
		case 2:
			return "third";
		case 3:
			return "fourth";
		default:
			return "!undefined!";
	}
}

document.addEventListener("DOMContentLoaded", () => {
	const startYear = 2023;
	let year = Date.prototype.getFullYear();

	let academicYear = year - startYear;

	let academicYearHasntEnded = Date.prototype.getMonth() < 5; // switch over in May

	if (academicYearHasntEnded) {
		academicYear -= 1;
	};

	document.getElementById("my-current-year").innerText = getYearName(academicYear);
});