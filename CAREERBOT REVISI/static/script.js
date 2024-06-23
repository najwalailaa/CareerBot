const swiper = new Swiper('.js-testimoni-slider', {
    grabCursor: true,
    spaceBetween: 30,
    pagination:{
        el:'.js-testimoni-pagination',
        clickable: true
    },
    breakpoints:{
        767:{
            slidesPerView: 2
        }
    }
});