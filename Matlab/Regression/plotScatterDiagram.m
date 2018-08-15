function [] = plotScatterDiagram(z_train, z_train_pred, z_test, z_test_pred, r_train, r_test)
    % Create a scatter diagram and label the axis 
    % save to a png file called scatter-m.png
    figure
    plot(z_train,z_train,'LineWidth',10)
    hold on
    scatter(z_train,z_train_pred,10,'og','filled')
    scatter(z_test,z_test_pred,10,'or','filled')
    hold off
    grid on
    legend_text={...
        ['1:1'],...
        ['Training Data (R ' num2str(r_train,2) ')'],...
        ['Testing Data (R ' num2str(r_test,2) ')']...
        };
    legend(legend_text,'Location','southeast');
    xlabel('Actual SPGScore','fontsize',20);
    ylabel('Estimated SPGScore','fontsize',20);
    title('Scatter Diagram','fontsize',25);
    set(gca,'FontSize',16)
    set(gca,'LineWidth',2);  
    print('scatter-m.png','-dpng')
end

