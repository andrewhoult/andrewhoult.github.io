class ProjectCard {
	constructor(title, description, link, imageUrl) {
		this.title = title;
		this.description = description;
		this.link = link;
		this.imageUrl = imageUrl;
	}

	getHTML() {
		return `
		<a href="${this.link}">
			<div class="project-card">
				<img src="${this.imageUrl}" alt="Project thumbnail"/>
				<div>
					<strong>${this.title}</strong>
					<p>${this.description}</p>
				</div>
			</div>
		</a>
		`
	}
}