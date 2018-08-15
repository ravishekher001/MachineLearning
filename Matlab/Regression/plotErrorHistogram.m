function [] = plotErrorHistogram(z_train, z_train_pred, z_test, z_test_pred)
    % Calculate the error in a variable called e and plot an error histogram using ploterrhist
    % save to a png file called nn-errhist-m.png
    e_train = z_train-z_train_pred;
    e_test = z_test-z_test_pred;
    figure
    ploterrhist(e_train,'Training',e_test,'Test')
    grid on
    set(gca,'FontSize',16)
    set(gca,'LineWidth',2);  
    print('errhist-m.png','-dpng')
end

