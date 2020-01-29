require 'uri'
require 'net/http'
require 'json'

class PonderationHooks < Redmine::Hook::ViewListener
    def controller_issues_new_before_save_after_qualification(context)
        setPonderation(context)
    end

    def setPonderation(context)
        customs = context[:params][:issue][:custom_field_values]
        weights = Setting.plugin_ponderation['weights']
        pond_field_id = Setting.plugin_ponderation['field_id']

        if !Project.find(context[:issue][:project_id]).enabled_module('auto ponderation') || !customs
            return nil
        end

        if weights
            ponderation = 0
            
            weights.each do |weight_field_id, value_coef_map|
                customs.each do |custom_field_id, custom_field_value|
                    if custom_field_id === weight_field_id
                        ponderation += value_coef_map[custom_field_value].to_f
                        break
                    end
                end
            end
            
            ponderation += Time.now.to_i / 86400
            context[:params][:issue][:custom_field_values][pond_field_id] = ponderation
        end
    end
end
