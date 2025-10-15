// Add shadow to navbar on scroll
document.addEventListener("scroll", () => {
  const nav = document.querySelector(".navbar");
  if (!nav) return;
  if (window.scrollY > 20) {
    nav.classList.add("scrolled");
  } else {
    nav.classList.remove("scrolled");
  }
});

// Glow effect for file input on upload page
document.addEventListener("DOMContentLoaded", () => {
  const fileInputs = document.querySelectorAll("input[type=file]");
  fileInputs.forEach(input => {
    input.addEventListener("change", () => {
      if (input.files.length > 0) {
        input.classList.add("btn-glow");
      } else {
        input.classList.remove("btn-glow");
      }
    });
  });
});

/* Main UI scripts */
// Navbar shadow on scroll (nice polish)
(function () {
  const nav = document.querySelector('.navbar');
  if (!nav) return;
  const onScroll = () => nav.classList.toggle('scrolled', window.scrollY > 4);
  onScroll(); window.addEventListener('scroll', onScroll);
})();

