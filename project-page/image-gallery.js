let images = new Array();

function addImage(url) {
	const img = document.createElement("img");
	img.src = url;

	document.getElementById("gallery-thumbnails").appendChild(img);

	if (images.length == 0) {
		document.getElementById("main-image").innerHTML = `<img src="${url}">`;
	}

	images.push(img);
}