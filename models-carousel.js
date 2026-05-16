// Models Carousel Script

const modelsData = [
    {
        emoji: '🍌',
        title: 'Banana Expert',
        description: 'Specialized CNN model for banana classification.',
        images: ['photos/bc.jpg', 'photos/bi.jpg', 'photos/br.jpg']
    },
    {
        emoji: '🥚',
        title: 'Egg Classifier',
        description: 'Optimized AI for egg contamination analysis.',
        images: ['photos/ec.jpg', 'photos/ei.jpg', 'photos/er.jpg']
    },
    {
        emoji: '🍊',
        title: 'Orange Pro',
        description: 'Advanced classification system for oranges.',
        images: ['photos/oc.jpg', 'photos/oi.jpg', 'photos/or.jpg']
    }
];

let currentModelIndex = 1; // Start with Egg Classifier

class ModelsCarousel {
    constructor() {
        this.init();
    }

    init() {
        // Wait for DOM elements to be available
        const timeout = setTimeout(() => {
            if (!this.centerEmoji) {
                console.warn('⚠️ Carousel elements not found. Make sure HTML structure is correct.');
                clearTimeout(timeout);
            }
        }, 1000);

        this.centerEmoji = document.getElementById('center-emoji');
        this.centerTitle = document.getElementById('center-title');
        this.centerDescription = document.getElementById('center-description');
        this.modelImages = document.querySelector('.model-images');
        this.dots = document.querySelectorAll('.dot');
        this.focusedCard = document.querySelector('.focused-card');
        this.sideModels = document.querySelectorAll('.side-model-item');

        if (!this.centerEmoji || !this.focusedCard) {
            console.error('❌ Carousel: Required DOM elements not found!');
            return;
        }

        console.log('✅ Carousel initialized');
        console.log('   Side models found:', this.sideModels.length);
        console.log('   Dots found:', this.dots.length);

        this.attachEventListeners();
        this.updateDisplay(currentModelIndex);
    }

    attachEventListeners() {
        // Dot navigation
        this.dots.forEach((dot, index) => {
            dot.addEventListener('click', () => {
                console.log('📍 Dot clicked:', index);
                this.switchModel(index);
            });
        });

        // Side model items click
        this.sideModels.forEach((item) => {
            item.addEventListener('click', () => {
                const dataIndex = parseInt(item.getAttribute('data-index'));
                console.log('👆 Side model clicked, index:', dataIndex);
                this.switchModel(dataIndex);
            });
        });

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') {
                console.log('⬅️ Arrow left pressed');
                this.switchModel(currentModelIndex - 1);
            }
            if (e.key === 'ArrowRight') {
                console.log('➡️ Arrow right pressed');
                this.switchModel(currentModelIndex + 1);
            }
        });
    }

    switchModel(newIndex) {
        // Wrap around
        if (newIndex < 0) newIndex = modelsData.length - 1;
        if (newIndex >= modelsData.length) newIndex = 0;

        if (newIndex === currentModelIndex) {
            return;
        }

        console.log('🔄 Switching from', modelsData[currentModelIndex].title, 'to', modelsData[newIndex].title);

        // Add transition out
        if (this.focusedCard) {
            this.focusedCard.classList.add('transition-out');
        }

        setTimeout(() => {
            currentModelIndex = newIndex;
            this.updateDisplay(newIndex);
            
            if (this.focusedCard) {
                this.focusedCard.classList.remove('transition-out');
                this.focusedCard.classList.add('transition-in');

                setTimeout(() => {
                    this.focusedCard.classList.remove('transition-in');
                }, 500);
            }
        }, 250);
    }

    updateDisplay(index) {
        const model = modelsData[index];

        console.log('📝 Updating display for:', model.title);

        // Update center card
        if (this.centerEmoji) this.centerEmoji.textContent = model.emoji;
        if (this.centerTitle) this.centerTitle.textContent = model.title;
        if (this.centerDescription) this.centerDescription.textContent = model.description;

        // Update images
        if (this.modelImages) {
            this.modelImages.innerHTML = model.images
                .map(img => `<img src="${img}" alt="${model.title} visualization" class="model-img">`)
                .join('');
        }

        // Update dots
        this.dots.forEach((dot, i) => {
            dot.classList.toggle('active', i === index);
        });

        // Update side models highlight
        this.sideModels.forEach((item) => {
            const itemIndex = parseInt(item.getAttribute('data-index'));
            item.classList.toggle('active', itemIndex === index);
        });
    }
}

// Initialize carousel when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('📦 DOM Content Loaded, initializing carousel...');
    new ModelsCarousel();
});