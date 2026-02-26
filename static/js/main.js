document.addEventListener("DOMContentLoaded", function () {
  const menuTrigger = document.getElementById("menu-trigger");
  const menuDropdown = document.getElementById("menu-dropdown");
  const fileInput = document.getElementById("file-input");
  const fileNameSpan = document.getElementById("file-name");
  const imageModeGroup = document.getElementById("image-mode-group");
  const defaultSubmitGroup = document.getElementById("default-submit-group");
  const userInput = document.getElementById("user-input");

  // If OCR extracted SMS text, prefill the textbox automatically.
  try {
    if (userInput && !userInput.value) {
      const prefill = sessionStorage.getItem("hfs_sms_prefill");
      if (prefill) {
        userInput.value = prefill;
        sessionStorage.removeItem("hfs_sms_prefill");
      }
    }
  } catch (e) {
    // ignore storage errors
  }

  if (menuTrigger && menuDropdown) {
    menuTrigger.addEventListener("click", function (e) {
      e.stopPropagation();
      menuDropdown.classList.toggle("show");
    });

    document.addEventListener("click", function () {
      menuDropdown.classList.remove("show");
    });
  }

  function isImageFile(filename) {
    const ext = filename.split(".").pop().toLowerCase();
    return ["png", "jpg", "jpeg", "gif"].includes(ext);
  }

  if (fileInput) {
    fileInput.addEventListener("change", function () {
      const file = fileInput.files[0];
      if (!file) {
        fileNameSpan.textContent = "";
        imageModeGroup.style.display = "none";
        defaultSubmitGroup.style.display = "block";
        return;
      }

      fileNameSpan.textContent = file.name;

      if (isImageFile(file.name)) {
        imageModeGroup.style.display = "block";
        defaultSubmitGroup.style.display = "none";
      } else {
        imageModeGroup.style.display = "none";
        defaultSubmitGroup.style.display = "block";
      }
    });
  }
});