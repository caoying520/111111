 if structure_method == "Xray":
                for times in range(10):
                    try:
                        if not os.path.exists(os.path.join(self.work_dir, r'{}.html'.format(Hit_id))):
                            print("  -->第{}尝试".format(times), flush=True)

                            Stoichiometry_response = requests.get(
                                url="https://www.rcsb.org/structure/{0}".format(Hit_id),
                                #proxies=proxies,
                                proxies=None,
                                headers={
                                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36",
                                    "Referer": "https://www.rcsb.org/",

                                }
                            )
                            if Stoichiometry_response.status_code == 200:
                                with open(os.path.join(self.work_dir, r'{}.html'.format(Hit_id)), "w", encoding="utf-8") as f_html:
                                    f_html.write(Stoichiometry_response.text)

                                Stoichiometry_soup = BeautifulSoup(Stoichiometry_response.text, "html.parser")

                                Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"carousel-footer"})
                                if Stoichiometry_text is None:
                                    Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"card-footer"})

                                Stoichiometry_info = \
                                Stoichiometry_text.text.split("Find")[0].split("Global Stoichiometry:")[1].strip()

                            else:
                                # 网络失败
                                print('*********')
                                Stoichiometry_info = ""
                        else:
                            with open(os.path.join(self.work_dir, r'{}.html'.format(Hit_id)), "r", encoding="utf-8") as f_html:
                                hitid_html = f_html.read()

                            Stoichiometry_soup = BeautifulSoup(hitid_html, "html.parser")

                            Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"carousel-footer"})
                            if Stoichiometry_text is None:
                                Stoichiometry_text = Stoichiometry_soup.find(name="div",attrs={"id":"Carousel-BiologicalUnit1"}).find(name="div",attrs={"class":"card-footer"})
                            Stoichiometry_info = Stoichiometry_text.text.split("Find")[0].split("Global Stoichiometry:")[1].strip()

                        try:
                            ligand_name_list = []
                            if crystal_info_dict["rcsb_entry_container_identifiers"]["non_polymer_entity_ids"]:
                                non_polymer_entity_ids = crystal_info_dict["rcsb_entry_container_identifiers"]["non_polymer_entity_ids"]

                                # ligand_name_list = crystal_info_dict["pdbx_vrpt_summary"]["restypes_notchecked_for_bond_angle_geometry"]
                                for non_polymer_entity_id in non_polymer_entity_ids:
                                    ligand_name = self.get_pdb_ligand_name(Hit_id, non_polymer_entity_id)
                                    if ligand_name:
                                        ligand_name_list.append(ligand_name)
                        except:
                            ligand_name_list = []


                        try:
                            # 兼容新旧RCSB JSON的Space Group字段
                            symmetry = crystal_info_dict.get("symmetry", {})
                            space_group = (symmetry.get("space_group_name_H_M") or symmetry.get("space_group_name_hm") or "")
                            resolution_list = crystal_info_dict.get(
                                "rcsb_entry_info", {}
                            ).get("resolution_combined", [])

                            resolution = str(resolution_list[0]) + " Å" if resolution_list else ""

                            cell_info = crystal_info_dict.get("cell", {})

                            table.cell(PDB_write_status, 0).text = Hit_id
                            table.cell(PDB_write_status, 1).text = space_group
                            table.cell(PDB_write_status, 2).text = resolution
                            table.cell(PDB_write_status, 3).text = (
                                str(cell_info.get("length_a", "")) + ", " +
                                str(cell_info.get("length_b", "")) + ", " +
                                str(cell_info.get("length_c", ""))
                            )
                            table.cell(PDB_write_status, 4).text = identities
                            table.cell(PDB_write_status, 5).text = ", ".join(ligand_name_list)
                            table.cell(PDB_write_status, 6).text = Stoichiometry_info

                            # 只有真正成功写入表格，才+1
                            Xray_count += 1

                            print("X-ray写入成功: {}，当前X-ray数量: {}".format(Hit_id, Xray_count),flush=True)
